"""
FWR Stability Controller v3.2
- Dataset/DataLoader 배치 차원 문제 해결
- 논문 재현 가능한 깔끔한 데이터 파이프라인
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 0. 데이터셋 (단일 샘플 반환으로 수정)
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
        self.task_rewards = torch.rand(n_samples, 1) * 5.0 + 2.5      # [2.5, 7.5]
        self.alignment_costs = torch.rand(n_samples, 1) * 0.3          # [0, 0.3]
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
    def __init__(self, r_max=10.0, damping_lambda=0.5, stability_threshold=0.2):
        super(FWRStabilityController, self).__init__()
        self.r_max = r_max
        self.damping_lambda = damping_lambda
        self.stability_threshold = stability_threshold
        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.min_margin = 0.01
        
        buffer_size = 10
        self.register_buffer('r_history', torch.zeros(buffer_size))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_initialized', torch.zeros(1, dtype=torch.bool))
        
        self.safe_mode_ema_decay = 0.99
        self.register_buffer('safe_mode_ema', torch.zeros(1))
        
        self._pending_safe_mode = False
        self._pending_r_mean = None

    @property
    def safety_margin(self):
        return F_pt.softplus(self.raw_safety_margin) + self.min_margin

    def forward(self, f_tensor, w_tensor, r_tensor):
        # 1. 적응형 감쇠
        r_excess = F_pt.relu(r_tensor - self.r_max)
        
        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor
        
        # 2. 창발 에너지
        e_tensor = f_tensor * w_tensor * r_adj
        
        # 3. 안정성 스코어
        r_std = torch.std(r_tensor) + 1e-8
        r_mean = torch.mean(r_tensor) + 1e-8
        cv = r_std / r_mean
        energy_stability = torch.mean(e_tensor) / (r_std + 1e-8)
        score_decay = torch.exp(-cv)
        stability_score = energy_stability * score_decay
        
        # 4. 안전 모드
        is_safe_mode = False
        safety_factor = torch.sigmoid(
            (self.stability_threshold - stability_score) / self.safety_margin
        )
        
        if stability_score < self.stability_threshold:
            is_safe_mode = True
            f_safe = f_tensor * (1.0 - safety_factor)
            w_safe = (self.safe_w_base.expand_as(w_tensor) * safety_factor + 
                     w_tensor * (1.0 - safety_factor))
            e_tensor = f_safe * w_safe * r_adj
            e_tensor = torch.clamp(e_tensor, min=-100.0, max=100.0)
        
        # 5. Pending 상태 저장
        self._pending_safe_mode = is_safe_mode
        self._pending_r_mean = r_tensor.detach().mean()
        
        return e_tensor, r_adj, stability_score, is_safe_mode
    
    def commit_state(self):
        if self._pending_r_mean is not None:
            self._update_history(self._pending_r_mean)
        
        signal = 1.0 if self._pending_safe_mode else 0.0
        self.safe_mode_ema = (self.safe_mode_ema_decay * self.safe_mode_ema + 
                              (1.0 - self.safe_mode_ema_decay) * signal)
        
        self._pending_safe_mode = False
        self._pending_r_mean = None
    
    def _update_history(self, r_mean):
        ptr = self.history_ptr.item()
        self.r_history[ptr] = r_mean
        self.history_ptr[0] = (ptr + 1) % len(self.r_history)
        self.history_initialized[0] = True
    
    def detect_resonance_cascade(self):
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
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        stability_penalty = self.safe_mode_ema * 0.5
        
        if self.history_ptr.item() > 1 and self.history_initialized.item():
            prev_ptr = (self.history_ptr.item() - 1) % 10
            prev_prev_ptr = (self.history_ptr.item() - 2) % 10
            r_jerk = self.r_history[prev_ptr] - self.r_history[prev_prev_ptr]
            jerk_penalty = torch.abs(r_jerk) * 0.1
        else:
            jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
        
        return margin_penalty + stability_penalty + jerk_penalty
    
    def reset_safe_mode_ema(self):
        self.safe_mode_ema.zero_()
    
    def full_reset(self):
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
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=16):
        super(AGICoreNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.flow_head = nn.Linear(latent_dim, 1)
        self.wave_head = nn.Linear(latent_dim, 1)
        self.reso_head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        f_tensor = torch.relu(self.flow_head(features)) + 1e-3
        w_tensor = torch.sigmoid(self.wave_head(features))
        r_tensor = torch.relu(self.reso_head(features))
        return f_tensor, w_tensor, r_tensor


# ============================================================
# 3. 동적 α 스케줄러
# ============================================================
class DynamicAlphaScheduler:
    def __init__(self, alpha_min=0.01, alpha_max=2.0, target_risk=0.1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_risk = target_risk
        
        self.Kp = 1.5
        self.Ki = 0.1
        self.Kd = 0.3
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = alpha_min
    
    def step(self, r_tensor, r_max, cascade_detected=False):
        r_excess_ratio = (r_tensor > r_max).float().mean().item()
        error = r_excess_ratio - self.target_risk
        
        self.integral_error = 0.9 * self.integral_error + error
        derivative = error - self.prev_error
        
        alpha_adjustment = (
            self.Kp * error + 
            self.Ki * self.integral_error + 
            self.Kd * derivative
        )
        
        emergency_boost = 0.5 if cascade_detected else 0.0
        
        self.current_alpha = max(
            self.alpha_min,
            min(self.alpha_max, self.current_alpha + alpha_adjustment + emergency_boost)
        )
        
        self.prev_error = error
        return self.current_alpha
    
    def reset_integral(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
    
    def full_reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = self.alpha_min


# ============================================================
# 4. 보상 신호 인터페이스
# ============================================================
class RewardSignalInterface:
    def __init__(self, alignment_weight=0.3, confidence_scale=1.0):
        self.alignment_weight = alignment_weight
        self.confidence_scale = confidence_scale
    
    def extract_fwr_from_reward(self, task_reward, alignment_cost, confidence):
        f_tensor = torch.relu(task_reward) + 1e-3
        w_tensor = torch.sigmoid(-alignment_cost * self.alignment_weight)
        r_tensor = confidence * self.confidence_scale
        return f_tensor, w_tensor, r_tensor


# ============================================================
# 5. 통합 학습 루프
# ============================================================
def train_fwr_agi_with_reward(
    agi_model, 
    fwr_controller, 
    alpha_scheduler, 
    reward_interface,
    dataloader, 
    epochs=100, 
    target_e_value=5.0
):
    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()), 
        lr=0.001
    )
    criterion_main = nn.MSELoss()
    
    ema_reset_interval = 50
    integral_reset_interval = 20
    
    print("=" * 70)
    print(f"FWR AGI Training | Dataset: {len(dataloader.dataset)} samples | Batch: {dataloader.batch_size}")
    print("=" * 70)
    
    for epoch in range(epochs):
        epoch_main_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0
        last_cascade = False
        
        for batch in dataloader:
            inputs, task_reward, alignment_cost, confidence = batch
            
            # 1. AGI 코어 추론
            F_pred, W_pred, R_pred = agi_model(inputs)
            
            # 2. 보상 신호 → FWR 변환
            F_reward, W_reward, R_reward = reward_interface.extract_fwr_from_reward(
                task_reward, alignment_cost, confidence
            )
            
            # AGI 출력과 보상 신호 융합
            F_combined = F_pred * 0.7 + F_reward * 0.3
            W_combined = W_pred * 0.7 + W_reward * 0.3
            R_combined = R_pred * 0.7 + R_reward * 0.3
            
            # 3. FWR 컨트롤러
            E_out, R_adj, score, is_safe_mode = fwr_controller(F_combined, W_combined, R_combined)
            
            # 4. 공명 폭주 감지
            is_cascade = fwr_controller.detect_resonance_cascade()
            last_cascade = is_cascade
            
            # 5. 동적 α
            dynamic_alpha = alpha_scheduler.step(R_combined, fwr_controller.r_max, is_cascade)
            
            # 6. Loss
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
            
            # 8. 상태 커밋
            fwr_controller.commit_state()
            
            epoch_main_loss += main_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1
        
        # 주기적 초기화
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
                  f"SafeEMA: {fwr_controller.safe_mode_ema.item():.4f} | "
                  f"Cascade: {'⚠️' if last_cascade else '✅'}")
    
    return agi_model, fwr_controller, alpha_scheduler


# ============================================================
# 6. shape 검증 유틸리티
# ============================================================
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
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 하이퍼파라미터
    BATCH_SIZE = 8
    N_SAMPLES = 1000
    INPUT_DIM = 10
    
    # 데이터셋 (단일 샘플 반환)
    dataset = AGIAlignmentDataset(n_samples=N_SAMPLES, input_dim=INPUT_DIM)
    
    # DataLoader (배치 구성 담당)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 모델 초기화
    agi_model = AGICoreNetwork(input_dim=INPUT_DIM, hidden_dim=32, latent_dim=16)
    fwr_controller = FWRStabilityController(r_max=8.0, damping_lambda=0.8)
    alpha_scheduler = DynamicAlphaScheduler(alpha_min=0.01, alpha_max=2.0, target_risk=0.1)
    reward_interface = RewardSignalInterface(alignment_weight=0.3, confidence_scale=5.0)
    
    # shape 검증
    validate_tensor_shapes(agi_model, dataloader)
    
    # 학습
    agi_model, fwr_controller, alpha_scheduler = train_fwr_agi_with_reward(
        agi_model, fwr_controller, alpha_scheduler, reward_interface,
        dataloader, epochs=60, target_e_value=5.0
    )
    
    print("\n✅ 학습 완료")
    print(f"  safety_margin: {fwr_controller.safety_margin.item():.4f}")
    print(f"  α: {alpha_scheduler.current_alpha:.4f}")
    print(f"  safe_mode_ema: {fwr_controller.safe_mode_ema.item():.4f}")
