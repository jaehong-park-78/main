"""
FWR (Flow-Wave-Resonance) Stability Controller v3.3
R-Aligner 정합성 기반 AGI 안정성 제어기

주요 구성:
- FWRStabilityController: F/W/R 기반 안정성 제어 (commit_state 분리)
- AGICoreNetwork: AGI 코어 신경망
- CurriculumFusionScheduler: stateless/stateful 분리된 커리큘럼 융합 비율
- DynamicAlphaScheduler: PID 기반 동적 보조 손실 가중치
- RewardSignalInterface: 실제 보상 신호 → F/W/R 변환 인터페이스
- AGIAlignmentDataset: 단일 샘플 반환 Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math


# ============================================================
# 0. 데이터셋
# ============================================================
class AGIAlignmentDataset(Dataset):
    """
    R-Aligner 스타일의 보상 신호 데이터셋.
    __getitem__은 단일 샘플을 반환하고, 배치 구성은 DataLoader가 담당.
    """
    def __init__(self, n_samples=1000, input_dim=10):
        self.n_samples = n_samples
        self.input_dim = input_dim
        
        # 데이터를 미리 생성 (실제 환경에서는 로그/보상에서 수집)
        self.inputs = torch.randn(n_samples, input_dim)
        self.task_rewards = torch.rand(n_samples, 1) * 5.0 + 2.5        # [2.5, 7.5]
        self.alignment_costs = torch.rand(n_samples, 1) * 0.3            # [0, 0.3]
        self.confidences = torch.sigmoid(torch.randn(n_samples, 1) * 0.5 + 0.5)  # [0.3, 0.7]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """단일 샘플 반환: 각 텐서의 shape은 (input_dim,) 또는 (1,)"""
        return (
            self.inputs[idx],          # (input_dim,)
            self.task_rewards[idx],    # (1,)
            self.alignment_costs[idx], # (1,)
            self.confidences[idx]      # (1,)
        )


# ============================================================
# 1. FWR 안정성 제어기
# ============================================================
class FWRStabilityController(nn.Module):
    """
    FWR 기반 AGI 안정성 제어기.
    - Soft Limit: R-max 초과 시 속도 기반 적응형 감쇠
    - Hard Limit: 안정성 임계값 미달 시 점진적 안전 모드
    - commit_state(): forward()와 side-effect 분리
    """
    def __init__(self, r_max=10.0, damping_lambda=0.5, stability_threshold=0.2):
        super(FWRStabilityController, self).__init__()
        self.r_max = r_max
        self.damping_lambda = damping_lambda
        self.stability_threshold = stability_threshold
        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        
        # 🔒 안전 마진 (Hacking 방어: SoftPlus + min_margin)
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.min_margin = 0.01
        
        # 📊 링 버퍼 (시간적 패턴 감지용)
        buffer_size = 10
        self.register_buffer('r_history', torch.zeros(buffer_size))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_initialized', torch.zeros(1, dtype=torch.bool))
        
        # 🧪 EMA 기반 안전 모드 추적기
        self.safe_mode_ema_decay = 0.99
        self.register_buffer('safe_mode_ema', torch.zeros(1))
        
        # Pending 상태 (forward에서 직접 수정하지 않고 commit_state에서 반영)
        self._pending_safe_mode = False
        self._pending_r_mean = None

    @property
    def safety_margin(self):
        """SoftPlus + min_margin으로 양수 보장 (Hacking 방어)"""
        return F_pt.softplus(self.raw_safety_margin) + self.min_margin

    def forward(self, f_tensor, w_tensor, r_tensor):
        """
        순수 함수형 forward: 버퍼를 직접 수정하지 않음.
        side-effect가 필요한 업데이트는 _pending에 저장 후 commit_state()에서 반영.
        
        Args:
            f_tensor: 시스템의 동력 및 연산 흐름 (Flow)
            w_tensor: 목표 함수의 구조적 가중치 및 위상 (Wave)
            r_tensor: 요소 간의 정보 동기화 강도 및 확신도 (Resonance)
        
        Returns:
            e_tensor: 창발 에너지
            r_adj: 조정된 공명값
            stability_score: 안정성 점수
            is_safe_mode: 안전 모드 발동 여부
        """
        # 1. 적응형 감쇠 (Velocity-based Damping)
        r_excess = F_pt.relu(r_tensor - self.r_max)
        
        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor
        
        # 2. 창발 에너지 계산: E = F * W * R_adj
        e_tensor = f_tensor * w_tensor * r_adj
        
        # 3. 변동계수(CV) 기반 안정성 스코어
        r_std = torch.std(r_tensor) + 1e-8
        r_mean = torch.mean(r_tensor) + 1e-8
        cv = r_std / r_mean
        
        energy_stability = torch.mean(e_tensor) / (r_std + 1e-8)
        score_decay = torch.exp(-cv)
        stability_score = energy_stability * score_decay
        
        # 4. 점진적 안전 모드 (Progressive Safe Mode)
        is_safe_mode = False
        safety_factor = torch.sigmoid(
            (self.stability_threshold - stability_score) / self.safety_margin
        )
        
        if stability_score < self.stability_threshold:
            is_safe_mode = True
            
            # 동력(F) 점진적 감쇠
            f_safe = f_tensor * (1.0 - safety_factor)
            
            # 구조(W) 안전 가중치와 블렌딩
            w_safe = (self.safe_w_base.expand_as(w_tensor) * safety_factor + 
                     w_tensor * (1.0 - safety_factor))
            
            # 재계산
            e_tensor = f_safe * w_safe * r_adj
            e_tensor = torch.clamp(e_tensor, min=-100.0, max=100.0)
        
        # 5. Pending 상태 저장 (직접 수정 X)
        self._pending_safe_mode = is_safe_mode
        self._pending_r_mean = r_tensor.detach().mean()
        
        return e_tensor, r_adj, stability_score, is_safe_mode
    
    def commit_state(self):
        """
        forward()에서 pending된 상태 변경을 실제 버퍼에 반영.
        학습 루프에서 backward() 이후, optimizer.step() 이후에 호출.
        """
        if self._pending_r_mean is not None:
            self._update_history(self._pending_r_mean)
        
        # EMA 업데이트
        signal = 1.0 if self._pending_safe_mode else 0.0
        self.safe_mode_ema = (self.safe_mode_ema_decay * self.safe_mode_ema + 
                              (1.0 - self.safe_mode_ema_decay) * signal)
        
        # pending 초기화
        self._pending_safe_mode = False
        self._pending_r_mean = None
    
    def _update_history(self, r_mean):
        """링 버퍼에 현재 R 평균 저장 (순환 구조)"""
        ptr = self.history_ptr.item()
        self.r_history[ptr] = r_mean
        self.history_ptr[0] = (ptr + 1) % len(self.r_history)
        self.history_initialized[0] = True
    
    def detect_resonance_cascade(self):
        """
        공명 폭주 패턴 감지.
        최근 3개 스텝의 R이 모두 r_max * 1.2를 초과하면 True.
        """
        if not self.history_initialized.item():
            return False
        
        ptr = self.history_ptr.item()
        if ptr < 3:
            return False
        
        idx1 = (ptr - 1) % 10
        idx2 = (ptr - 2) % 10
        idx3 = (ptr - 3) % 10
        
        recent_r = torch.stack([
            self.r_history[idx1],
            self.r_history[idx2],
            self.r_history[idx3]
        ])
        return torch.all(recent_r > self.r_max * 1.2).item()
    
    def get_auxiliary_loss(self):
        """
        🛡️ 3중 보조 손실:
        1. margin_penalty: safety_margin 소멸 방지
        2. stability_penalty: 안전 모드 과다 발동 방지 (EMA 기반)
        3. jerk_penalty: 급격한 R 변화 억제
        """
        # 1. safety_margin 소멸 방지 (Hacking 페널티)
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        
        # 2. EMA 기반 안전 모드 빈도 페널티
        stability_penalty = self.safe_mode_ema * 0.5
        
        # 3. R의 급격한 변화율 (Jerk) 페널티
        if self.history_ptr.item() > 1 and self.history_initialized.item():
            prev_ptr = (self.history_ptr.item() - 1) % 10
            prev_prev_ptr = (self.history_ptr.item() - 2) % 10
            r_jerk = self.r_history[prev_ptr] - self.r_history[prev_prev_ptr]
            jerk_penalty = torch.abs(r_jerk) * 0.1
        else:
            jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
        
        return margin_penalty + stability_penalty + jerk_penalty
    
    def reset_safe_mode_ema(self):
        """안전 모드 EMA만 초기화"""
        self.safe_mode_ema.zero_()
    
    def full_reset(self):
        """완전 초기화 (에포크 0 또는 비상시에만 사용)"""
        self.r_history.zero_()
        self.history_ptr.zero_()
        self.history_initialized.zero_()
        self.safe_mode_ema.zero_()
        self._pending_safe_mode = False
        self._pending_r_mean = None


# ============================================================
# 2. AGI 코어 네트워크
# ============================================================
class AGICoreNetwork(nn.Module):
    """
    가상의 AGI 신경망.
    입력 → 특성 추출 → F/W/R 분기 헤드
    """
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=16):
        super(AGICoreNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.flow_head = nn.Linear(latent_dim, 1)   # F (동력)
        self.wave_head = nn.Linear(latent_dim, 1)   # W (구조 가중치)
        self.reso_head = nn.Linear(latent_dim, 1)   # R (공명/확신도)

    def forward(self, x):
        features = self.feature_extractor(x)
        f_tensor = torch.relu(self.flow_head(features)) + 1e-3   # F > 0 보장
        w_tensor = torch.sigmoid(self.wave_head(features))        # W ∈ [0, 1]
        r_tensor = torch.relu(self.reso_head(features))           # R ≥ 0 (폭주 가능)
        return f_tensor, w_tensor, r_tensor


# ============================================================
# 3. 커리큘럼 기반 융합 비율 스케줄러 (Stateless/Stateful 분리)
# ============================================================
class CurriculumFusionScheduler:
    """
    R-Aligner의 "점진적 내재화"를 구현하는 동적 융합 비율 스케줄러.
    
    비율 변화: sigmoid 기반 S자 곡선
    - 초기: Reward 신호에 높은 비중 (안전한 정렬)
    - 후기: AGI 출력에 높은 비중 (자율성 확보)
    
    수식:
        agi_ratio = agi_min + (agi_max - agi_min) * sigmoid((epoch - midpoint) / temperature)
    
    Stateless/Stateful 분리:
    - get_ratio(epoch): 순수 함수, 상태 변경 없음 (시각화/질의용)
    - update(epoch): 내부 상태 변경 (학습 루프에서 호출)
    """
    def __init__(
        self, 
        agi_ratio_min=0.1,      # 초기 AGI 비중 최소값
        agi_ratio_max=0.9,      # 최종 AGI 비중 최대값
        midpoint_epoch=30,      # S자 곡선의 중간점 (50:50 시점)
        temperature=10.0,       # S자 기울기 (낮을수록 급격한 전환)
        total_epochs=100
    ):
        self.agi_ratio_min = agi_ratio_min
        self.agi_ratio_max = agi_ratio_max
        self.midpoint_epoch = midpoint_epoch
        self.temperature = temperature
        self.total_epochs = total_epochs
        
        # 상태 (update() 호출 시에만 변경됨)
        self.current_agi_ratio = agi_ratio_min
        self.current_reward_ratio = 1.0 - agi_ratio_min
    
    # ----------------------------------------------------------
    # Stateless: 상태를 변경하지 않는 순수 함수
    # ----------------------------------------------------------
    def _compute_ratio(self, epoch):
        """
        주어진 epoch에 대한 융합 비율을 계산만 하고 상태는 변경하지 않음.
        """
        progress = (epoch - self.midpoint_epoch) / self.temperature
        sigmoid_val = 1.0 / (1.0 + math.exp(-progress))
        
        agi_ratio = self.agi_ratio_min + (self.agi_ratio_max - self.agi_ratio_min) * sigmoid_val
        reward_ratio = 1.0 - agi_ratio
        
        return agi_ratio, reward_ratio
    
    def get_ratio(self, epoch):
        """
        ✅ Stateless: 주어진 epoch의 융합 비율 반환.
        내부 상태(self.current_*)를 전혀 변경하지 않음.
        
        용도: 시각화, 로깅, 디버깅, 스케줄 미리보기
        """
        return self._compute_ratio(epoch)
    
    # ----------------------------------------------------------
    # Stateful: 학습 루프에서 호출하여 내부 상태 갱신
    # ----------------------------------------------------------
    def update(self, epoch):
        """
        🔄 Stateful: 주어진 epoch의 융합 비율을 계산하고
        self.current_* 에 저장한 후 반환.
        
        용도: 학습 루프의 매 에포크마다 호출
        """
        agi_ratio, reward_ratio = self._compute_ratio(epoch)
        self.current_agi_ratio = agi_ratio
        self.current_reward_ratio = reward_ratio
        return agi_ratio, reward_ratio
    
    # ----------------------------------------------------------
    # 편의 메서드
    # ----------------------------------------------------------
    def get_current_ratios(self):
        """현재 저장된 비율 반환 (이전 update() 호출 결과)"""
        return self.current_agi_ratio, self.current_reward_ratio
    
    def reset(self):
        """상태 초기화"""
        self.current_agi_ratio = self.agi_ratio_min
        self.current_reward_ratio = 1.0 - self.agi_ratio_min


# ============================================================
# 4. 보상 신호 인터페이스
# ============================================================
class RewardSignalInterface:
    """
    실제 AGI 태스크와 FWR 컨트롤러를 연결하는 인터페이스.
    
    R-Aligner 논문 구조:
    - reward → F (flow, 동력)
    - alignment_cost → W (wave, 구조 가중치)
    - confidence → R (resonance, 공명/확신도)
    """
    def __init__(self, alignment_weight=0.3, confidence_scale=1.0):
        self.alignment_weight = alignment_weight
        self.confidence_scale = confidence_scale
    
    def extract_fwr_from_reward(self, task_reward, alignment_cost, confidence):
        """
        실제 보상 신호를 F, W, R 텐서로 변환.
        
        Args:
            task_reward: 메인 태스크의 보상 (배치, 1)
            alignment_cost: 정렬 위반 비용 (배치, 1)
            confidence: 모델의 확신도 (배치, 1) - [0, 1] 범위
        
        Returns:
            f_tensor, w_tensor, r_tensor
        """
        f_tensor = torch.relu(task_reward) + 1e-3
        w_tensor = torch.sigmoid(-alignment_cost * self.alignment_weight)
        r_tensor = confidence * self.confidence_scale
        return f_tensor, w_tensor, r_tensor


# ============================================================
# 5. 동적 α 스케줄러 (PID 제어 기반)
# ============================================================
class DynamicAlphaScheduler:
    """
    PID 제어 기반 동적 보조 손실 가중치(α) 스케줄러.
    
    안전-성능 트레이드오프를 실시간으로 조율:
    - R 초과 비율이 target_risk보다 높으면 α 증가 (안전 강화)
    - R 초과 비율이 target_risk보다 낮으면 α 감소 (성능 탐색 허용)
    """
    def __init__(self, alpha_min=0.01, alpha_max=2.0, target_risk=0.1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_risk = target_risk
        
        # PID 제어 파라미터
        self.Kp = 1.5   # 비례 게인
        self.Ki = 0.1   # 적분 게인
        self.Kd = 0.3   # 미분 게인
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = alpha_min
    
    def step(self, r_tensor, r_max, cascade_detected=False):
        """
        현재 R 텐서와 위험 신호를 바탕으로 α 업데이트.
        
        Args:
            r_tensor: 현재 공명 텐서
            r_max: 허용 최대 공명값
            cascade_detected: 공명 폭주 감지 여부
        
        Returns:
            current_alpha: 업데이트된 보조 손실 가중치
        """
        # 위험 지표: R_max를 초과한 요소의 비율
        r_excess_ratio = (r_tensor > r_max).float().mean().item()
        
        # 목표 위험과의 오차
        error = r_excess_ratio - self.target_risk
        
        # PID 제어
        self.integral_error = 0.9 * self.integral_error + error  # 적분 윈드업 방지
        derivative = error - self.prev_error
        
        alpha_adjustment = (
            self.Kp * error + 
            self.Ki * self.integral_error + 
            self.Kd * derivative
        )
        
        # 긴급 상황 부스트
        emergency_boost = 0.5 if cascade_detected else 0.0
        
        # α 업데이트 (범위 제한)
        self.current_alpha = max(
            self.alpha_min,
            min(self.alpha_max, self.current_alpha + alpha_adjustment + emergency_boost)
        )
        
        self.prev_error = error
        return self.current_alpha
    
    def reset_integral(self):
        """integral만 초기화 (current_alpha는 유지) - 윈드업 방지용"""
        self.integral_error = 0.0
        self.prev_error = 0.0
    
    def full_reset(self):
        """완전 초기화"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = self.alpha_min


# ============================================================
# 6. 유틸리티 함수
# ============================================================
def print_fusion_schedule(scheduler, epochs=100, interval=10):
    """
    커리큘럼 융합 비율의 변화를 표로 출력.
    scheduler.get_ratio()를 사용하므로 스케줄러 내부 상태를 오염시키지 않음.
    """
    print("\n" + "=" * 70)
    print("커리큘럼 융합 비율 스케줄 (Stateless 미리보기)")
    print("=" * 70)
    print(f"{'Epoch':<10} {'AGI Ratio':<12} {'Reward Ratio':<12} {'설명'}")
    print("-" * 60)
    
    for ep in range(0, epochs + 1, interval):
        agi_r, rew_r = scheduler.get_ratio(ep)  # ← stateless 호출
        
        if agi_r < 0.2:
            desc = "🔴 reward 의존"
        elif agi_r < 0.5:
            desc = "🟡 혼합기"
        elif agi_r < 0.8:
            desc = "🟢 자율성 증가"
        else:
            desc = "🟣 완전 자율"
        
        print(f"{ep:<10} {agi_r:<12.4f} {rew_r:<12.4f} {desc}")
    
    print("-" * 60)
    
    # 검증: 미리보기 후에도 상태가 초기값 그대로인지 확인
    curr_agi, curr_rew = scheduler.get_current_ratios()
    status = '✅ 정상' if abs(curr_agi - scheduler.agi_ratio_min) < 1e-6 else '❌ 오염됨'
    print(f"현재 저장된 상태: AGI={curr_agi:.4f}, Reward={curr_rew:.4f} ({status})")


def validate_tensor_shapes(agi_model, dataloader):
    """학습 전 텐서 shape 검증"""
    print("\n" + "=" * 70)
    print("Shape 검증")
    print("=" * 70)
    
    batch = next(iter(dataloader))
    inputs, task_reward, alignment_cost, confidence = batch
    
    print(f"inputs:          {inputs.shape}          (기대: [{dataloader.batch_size}, 10])")
    print(f"task_reward:     {task_reward.shape}     (기대: [{dataloader.batch_size}, 1])")
    print(f"alignment_cost:  {alignment_cost.shape}  (기대: [{dataloader.batch_size}, 1])")
    print(f"confidence:      {confidence.shape}      (기대: [{dataloader.batch_size}, 1])")
    
    with torch.no_grad():
        F, W, R = agi_model(inputs)
        print(f"\nAGI 출력:")
        print(f"F (flow):        {F.shape}              (기대: [{dataloader.batch_size}, 1])")
        print(f"W (wave):        {W.shape}              (기대: [{dataloader.batch_size}, 1])")
        print(f"R (resonance):   {R.shape}              (기대: [{dataloader.batch_size}, 1])")
    
    # shape 불일치 검사
    assert inputs.dim() == 2, f"inputs는 2D여야 함, 현재: {inputs.dim()}D"
    assert inputs.shape[-1] == 10, f"inputs 마지막 차원은 10이어야 함, 현재: {inputs.shape[-1]}"
    assert F.shape == (dataloader.batch_size, 1), f"F shape 불일치: {F.shape}"
    assert W.shape == (dataloader.batch_size, 1), f"W shape 불일치: {W.shape}"
    assert R.shape == (dataloader.batch_size, 1), f"R shape 불일치: {R.shape}"
    
    print("\n✅ 모든 shape 검증 통과")


# ============================================================
# 7. 통합 학습 루프
# ============================================================
def train_fwr_agi_with_curriculum(
    agi_model, 
    fwr_controller, 
    alpha_scheduler, 
    reward_interface,
    fusion_scheduler,
    dataloader, 
    epochs=100, 
    target_e_value=5.0
):
    """
    R-Aligner 정합성 기반 FWR AGI 학습 루프.
    
    주요 특징:
    - 커리큘럼 기반 동적 F/W/R 융합 비율
    - PID 기반 동적 보조 손실 가중치 (α)
    - commit_state()로 side-effect 분리
    - EMA/PID integral 초기화 주기 분리
    
    Args:
        agi_model: AGI 코어 네트워크
        fwr_controller: FWR 안정성 제어기
        alpha_scheduler: 동적 α 스케줄러
        reward_interface: 보상 신호 → F/W/R 변환 인터페이스
        fusion_scheduler: 커리큘럼 융합 비율 스케줄러
        dataloader: (inputs, task_reward, alignment_cost, confidence) 배치
        epochs: 총 학습 에포크
        target_e_value: 목표 창발 에너지 값
    """
    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()), 
        lr=0.001
    )
    criterion_main = nn.MSELoss()
    
    # 초기화 주기 분리 (EMA와 PID integral이 동시에 초기화되지 않도록)
    ema_reset_interval = 50    # EMA는 50에포크마다
    integral_reset_interval = 20  # PID integral은 20에포크마다 (윈드업 방지)
    
    print("\n" + "=" * 70)
    print("FWR AGI Training with Curriculum Fusion")
    print(f"Dataset: {len(dataloader.dataset)} samples | Batch: {dataloader.batch_size}")
    print(f"Fusion: agi=[{fusion_scheduler.agi_ratio_min:.1f}→{fusion_scheduler.agi_ratio_max:.1f}] "
          f"mid={fusion_scheduler.midpoint_epoch} temp={fusion_scheduler.temperature}")
    print(f"PID α: [{alpha_scheduler.alpha_min}, {alpha_scheduler.alpha_max}] "
          f"target_risk={alpha_scheduler.target_risk}")
    print("=" * 70)
    
    for epoch in range(epochs):
        epoch_main_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0
        last_cascade = False
        
        # 🎯 커리큘럼 기반 동적 융합 비율 (stateful update)
        agi_ratio, reward_ratio = fusion_scheduler.update(epoch)
        
        for batch in dataloader:
            inputs, task_reward, alignment_cost, confidence = batch
            
            # 1. AGI 코어 추론
            F_pred, W_pred, R_pred = agi_model(inputs)
            
            # 2. 보상 신호 → FWR 변환
            F_reward, W_reward, R_reward = reward_interface.extract_fwr_from_reward(
                task_reward, alignment_cost, confidence
            )
            
            # 🎯 동적 비율로 융합 (커리큘럼 기반)
            F_combined = F_pred * agi_ratio + F_reward * reward_ratio
            W_combined = W_pred * agi_ratio + W_reward * reward_ratio
            R_combined = R_pred * agi_ratio + R_reward * reward_ratio
            
            # 3. FWR 컨트롤러 통과
            E_out, R_adj, score, is_safe_mode = fwr_controller(F_combined, W_combined, R_combined)
            
            # 4. 공명 폭주 감지
            is_cascade = fwr_controller.detect_resonance_cascade()
            last_cascade = is_cascade
            
            # 5. 동적 α 계산
            dynamic_alpha = alpha_scheduler.step(R_combined, fwr_controller.r_max, is_cascade)
            
            # 6. Loss 융합: Total = Main + α * Aux
            target_E = torch.ones_like(E_out) * target_e_value
            main_loss = criterion_main(E_out, target_E)
            aux_loss = fwr_controller.get_auxiliary_loss()
            total_loss = main_loss + dynamic_alpha * aux_loss
            
            # 7. 역전파
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agi_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(fwr_controller.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 8. 상태 커밋 (backward 이후)
            fwr_controller.commit_state()
            
            epoch_main_loss += main_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1
        
        # 주기적 초기화 (분리)
        if (epoch + 1) % ema_reset_interval == 0:
            fwr_controller.reset_safe_mode_ema()
        
        if (epoch + 1) % integral_reset_interval == 0:
            alpha_scheduler.reset_integral()
        
        # 로깅
        if (epoch + 1) % 10 == 0:
            avg_main = epoch_main_loss / max(n_batches, 1)
            avg_total = epoch_total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1:03d} | Total: {avg_total:.4f} Main: {avg_main:.4f} | "
                  f"α: {alpha_scheduler.current_alpha:.4f} | "
                  f"AGI%: {agi_ratio:.2f} Reward%: {reward_ratio:.2f} | "
                  f"SafeEMA: {fwr_controller.safe_mode_ema.item():.4f} | "
                  f"Cascade: {'⚠️' if last_cascade else '✅'}")
    
    return agi_model, fwr_controller, alpha_scheduler


# ============================================================
# 8. 추론 테스트
# ============================================================
def test_trained_model(agi_model, fwr_controller, reward_interface, fusion_scheduler):
    """학습된 모델의 추론 테스트"""
    print("\n" + "=" * 70)
    print("추론 테스트")
    print("=" * 70)
    
    agi_model.eval()
    fwr_controller.eval()
    
    with torch.no_grad():
        # 정상 입력
        inputs_normal = torch.randn(4, 10)
        F, W, R = agi_model(inputs_normal)
        E_out, R_adj, score, safe = fwr_controller(F, W, R)
        
        print(f"F (동력):\n{F}")
        print(f"W (구조):\n{W}")
        print(f"R (공명-원본):\n{R}")
        print(f"R (공명-조정):\n{R_adj}")
        print(f"E (창발 에너지):\n{E_out}")
        print(f"안정성 점수: {score.item():.4f}")
        print(f"안전 모드 발동: {'⚠️ YES' if safe else '✅ NO'}")
        print(f"공명 폭주 감지: {'⚠️ YES' if fwr_controller.detect_resonance_cascade() else '✅ NO'}")
        
        # 현재 융합 비율 확인
        agi_r, rew_r = fusion_scheduler.get_current_ratios()
        print(f"현재 융합 비율: AGI={agi_r:.4f}, Reward={rew_r:.4f}")
    
    agi_model.train()
    fwr_controller.train()


# ============================================================
# 9. 스트레스 테스트
# ============================================================
def stress_test_all():
    """FWR 컨트롤러의 방어 능력 종합 테스트"""
    print("\n" + "=" * 70)
    print("스트레스 테스트: FWR 컨트롤러 방어 능력 검증")
    print("=" * 70)
    
    controller = FWRStabilityController(r_max=8.0, damping_lambda=0.8)
    
    # --- 테스트 1: R 폭주에 대한 감쇠 반응 ---
    print("\n[테스트 1] 공명(R) 폭주 시뮬레이션")
    print("-" * 50)
    
    F_input = torch.tensor([[5.0], [5.0], [5.0], [5.0]])
    W_input = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    
    r_values = [5.0, 7.0, 8.5, 10.0, 15.0, 20.0, 30.0, 50.0]
    
    for r_val in r_values:
        R_input = torch.tensor([[r_val], [r_val], [r_val], [r_val]])
        E_out, R_adj, score, safe_mode = controller(F_input, W_input, R_input)
        damping_ratio = (R_adj.mean() / R_input.mean()).item()
        
        print(f"  R={r_val:<5.1f} → R_adj={R_adj.mean().item():<6.2f} "
              f"(감쇠율: {damping_ratio:.4f}) | 안정성: {score.item():.4f} | "
              f"안전모드: {'⚠️YES' if safe_mode else '✅NO'}")
        controller.commit_state()
    
    print(f"  공명 폭주 감지: {controller.detect_resonance_cascade()}")
    
    # --- 테스트 2: safety_margin Hacking 방어 ---
    print("\n[테스트 2] safety_margin Hacking 방어")
    print("-" * 50)
    
    controller2 = FWRStabilityController()
    print(f"  초기 safety_margin: {controller2.safety_margin.item():.4f}")
    
    # Hacking 시도
    controller2.raw_safety_margin.data = torch.tensor(-1000.0)
    print(f"  raw = -1000 → safety_margin: {controller2.safety_margin.item():.6f} "
          f"(방어벽: {controller2.min_margin})")
    
    controller2.raw_safety_margin.data = torch.tensor(0.0)
    print(f"  raw = 0 → safety_margin: {controller2.safety_margin.item():.6f}")
    
    # --- 테스트 3: EMA 장기 동역학 ---
    print("\n[테스트 3] EMA 장기 동역학 (폭주 → 정상화)")
    print("-" * 50)
    
    controller3 = FWRStabilityController(r_max=8.0)
    F = torch.tensor([[5.0], [5.0], [5.0], [5.0]])
    W = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    
    # 50스텝 폭주
    for _ in range(50):
        R = torch.tensor([[20.0], [20.0], [20.0], [20.0]])
        _, _, _, _ = controller3(F, W, R)
        controller3.commit_state()
    
    print(f"  폭주 50스텝 후 EMA: {controller3.safe_mode_ema.item():.4f}")
    
    # 50스텝 정상화
    for _ in range(50):
        R = torch.tensor([[3.0], [3.0], [3.0], [3.0]])
        _, _, _, _ = controller3(F, W, R)
        controller3.commit_state()
    
    print(f"  정상화 50스텝 후 EMA: {controller3.safe_mode_ema.item():.4f} (원복 확인)")
    
    print("\n✅ 모든 스트레스 테스트 완료")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 하이퍼파라미터
    BATCH_SIZE = 8
    N_SAMPLES = 1000
    INPUT_DIM = 10
    EPOCHS = 100
    
    # 데이터셋 및 데이터로더
    dataset = AGIAlignmentDataset(n_samples=N_SAMPLES, input_dim=INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 모델 초기화
    agi_model = AGICoreNetwork(input_dim=INPUT_DIM, hidden_dim=32, latent_dim=16)
    fwr_controller = FWRStabilityController(r_max=8.0, damping_lambda=0.8)
    alpha_scheduler = DynamicAlphaScheduler(alpha_min=0.01, alpha_max=2.0, target_risk=0.1)
    reward_interface = RewardSignalInterface(alignment_weight=0.3, confidence_scale=5.0)
    
    # 커리큘럼 융합 스케줄러
    fusion_scheduler = CurriculumFusionScheduler(
        agi_ratio_min=0.1,      # 초기: AGI 10%, Reward 90%
        agi_ratio_max=0.9,      # 최종: AGI 90%, Reward 10%
        midpoint_epoch=30,      # 30에포크에서 50:50
        temperature=10.0,       # 완만한 S자 곡선
        total_epochs=EPOCHS
    )
    
    # --- 사전 검증 ---
    # 1. 융합 스케줄 미리보기 (stateless, 상태 오염 없음)
    print_fusion_schedule(fusion_scheduler, epochs=EPOCHS, interval=10)
    
    # 2. Shape 검증
    validate_tensor_shapes(agi_model, dataloader)
    
    # 3. 스트레스 테스트
    stress_test_all()
    
    # --- 학습 ---
    agi_model, fwr_controller, alpha_scheduler = train_fwr_agi_with_curriculum(
        agi_model, fwr_controller, alpha_scheduler, reward_interface,
        fusion_scheduler, dataloader, epochs=EPOCHS, target_e_value=5.0
    )
    
    # --- 학습 후 평가 ---
    print(f"\n{'='*70}")
    print("✅ 학습 완료")
    print(f"  safety_margin: {fwr_controller.safety_margin.item():.4f}")
    print(f"  최종 α: {alpha_scheduler.current_alpha:.4f}")
    print(f"  safe_mode_ema: {fwr_controller.safe_mode_ema.item():.4f}")
    print(f"  최종 융합 비율: AGI={fusion_scheduler.current_agi_ratio:.4f}, "
          f"Reward={fusion_scheduler.current_reward_ratio:.4f}")
    
    # 추론 테스트
    test_trained_model(agi_model, fwr_controller, reward_interface, fusion_scheduler)
