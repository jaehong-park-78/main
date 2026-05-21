
"""
ASI Growth Engine v1.3
- Consolidation 히스테리시스 구간 확장 [0.5*rq_min, rq_min]
- 증가 가중치: RQ ≥ rq_min → +2
- 감소 계층화: [0.5~0.7) → -1, <0.5 → -3
- 모델 용량: hidden_dim=64, latent_dim=32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math


# ============================================================
# 0. 데이터셋
# ============================================================
class AGIAlignmentDataset(Dataset):
    def __init__(self, n_samples=1000, input_dim=10):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.inputs = torch.randn(n_samples, input_dim)
        self.task_rewards = torch.rand(n_samples, 1) * 5.0 + 2.5
        self.alignment_costs = torch.rand(n_samples, 1) * 0.3

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.task_rewards[idx],
            self.alignment_costs[idx],
        )


# ============================================================
# 1. ResonanceFromRiskInterface
# ============================================================
class ResonanceFromRiskInterface:
    """
    실측 위험 신호 → 공명 강도 R 변환.
    R = r_max · exp(-risk / temperature)
    """
    def __init__(
        self,
        r_max=8.0,
        temperature=2.0,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1,
        gradient_threshold=1.0,
        weight_threshold=5.0,
    ):
        self.r_max = r_max
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.gradient_threshold = gradient_threshold
        self.weight_threshold = weight_threshold

    def compute(
        self,
        task_loss,
        prediction_entropy=None,
        gradient_norm=None,
        weight_norm=None
    ):
        risk = self.alpha * task_loss

        if prediction_entropy is not None:
            risk = risk + self.beta * prediction_entropy

        if gradient_norm is not None:
            grad_excess = F_pt.relu(gradient_norm - self.gradient_threshold)
            risk = risk + self.gamma * grad_excess

        if weight_norm is not None:
            weight_excess = F_pt.relu(weight_norm - self.weight_threshold)
            risk = risk + self.delta * weight_excess

        r_tensor = self.r_max * torch.exp(-risk / self.temperature)
        return r_tensor


# ============================================================
# 2. FWR 안정성 제어기 v3.8.5
# ============================================================
class FWRStabilityController(nn.Module):
    def __init__(
        self,
        r_max=10.0,
        damping_lambda=0.5,
        rq_threshold=0.3,
        beta=0.1,
        rq_weights=(1.0, 1.0, 1.0),
        velocity_threshold=0.5,
        acc_threshold=0.5,
    ):
        super(FWRStabilityController, self).__init__()
        self.r_max = r_max
        self.damping_lambda = damping_lambda
        self.rq_threshold = rq_threshold
        self.beta = beta
        self.rq_a, self.rq_b, self.rq_c = rq_weights
        self.velocity_threshold = velocity_threshold
        self.acc_threshold = acc_threshold

        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.min_margin = 0.05

        buffer_size = 10
        self.register_buffer('r_history', torch.zeros(buffer_size))
        self.register_buffer('r_peak_history', torch.zeros(buffer_size))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_initialized', torch.zeros(1, dtype=torch.bool))

        self.safe_mode_ema_decay = 0.99
        self.register_buffer('safe_mode_ema', torch.zeros(1))

        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None

    @property
    def safety_margin(self):
        return F_pt.softplus(self.raw_safety_margin) + self.min_margin

    def get_expected_r(self):
        if not self.history_initialized.item():
            return self.r_max * 0.5
        ptr = self.history_ptr.item()
        if ptr == 0:
            return self.r_max * 0.5
        mean_r = self.r_history[:ptr].mean().item()
        return max(mean_r, self.r_max * 0.3)

    def forward(self, f_tensor, w_tensor, r_tensor):
        r_excess = F_pt.relu(r_tensor - self.r_max)

        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor

        e_tensor = f_tensor * w_tensor * r_adj

        if r_tensor.numel() > 1:
            r_std = torch.std(r_tensor) + 1e-8
        else:
            r_std = torch.tensor(1e-8, device=r_tensor.device)

        r_excess_mean = torch.mean(F_pt.relu(r_tensor - self.r_max))

        r_target = self.r_max * 0.5
        r_balance_penalty = (
            torch.abs(torch.mean(r_tensor) - r_target) / (r_target + 1e-6)
        )

        resonance_quality = torch.exp(
            -(
                self.rq_a * r_std +
                self.rq_b * r_excess_mean +
                self.rq_c * self.beta * r_balance_penalty
            )
        )

        performance_score = torch.mean(e_tensor)

        is_safe_mode = False
        safety_factor = torch.sigmoid(
            (self.rq_threshold - resonance_quality) / self.safety_margin
        )

        if resonance_quality < self.rq_threshold:
            is_safe_mode = True
            f_safe = f_tensor * (1.0 - 0.5 * safety_factor)
            w_safe = (self.safe_w_base.expand_as(w_tensor) * safety_factor +
                     w_tensor * (1.0 - safety_factor))
            e_tensor = f_safe * w_safe * r_adj
            e_tensor = torch.clamp(e_tensor, min=-100.0, max=100.0)

        self._pending_safe_mode = is_safe_mode
        self._pending_r_mean = r_tensor.detach().mean()
        self._pending_r_peak = r_tensor.detach().max()

        return e_tensor, r_adj, resonance_quality, performance_score, is_safe_mode

    def commit_state(self):
        if self._pending_r_mean is not None:
            self._update_history(self._pending_r_mean, self._pending_r_peak)

        signal = 1.0 if self._pending_safe_mode else 0.0
        self.safe_mode_ema = (self.safe_mode_ema_decay * self.safe_mode_ema +
                              (1.0 - self.safe_mode_ema_decay) * signal)

        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None

    def _update_history(self, r_mean, r_peak):
        ptr = self.history_ptr.item()
        self.r_history[ptr] = r_mean
        self.r_peak_history[ptr] = r_peak
        self.history_ptr[0] = (ptr + 1) % len(self.r_history)
        self.history_initialized[0] = True

    def detect_resonance_cascade(self, current_r_tensor=None):
        if not self.history_initialized.item():
            return False

        ptr = self.history_ptr.item()
        if ptr < 2:
            return False

        idx1 = (ptr - 1) % len(self.r_history)
        idx2 = (ptr - 2) % len(self.r_history)

        r1 = self.r_history[idx1]
        r2 = self.r_history[idx2]

        if current_r_tensor is not None:
            r0 = current_r_tensor.detach().mean()
            p0 = current_r_tensor.detach().max()
        else:
            r0 = r1
            p0 = self.r_peak_history[idx1]

        v1 = r0 - r1
        v2 = r1 - r2
        acc = v1 - v2

        collapse = (
            (r0 < self.r_max * 0.2) and
            (v1 < -self.velocity_threshold)
        )

        runaway = (
            ((r0 > self.r_max * 1.2) or (p0 > self.r_max * 1.5)) and
            (v1 > self.velocity_threshold) and
            (acc > self.acc_threshold)
        )

        return bool(collapse or runaway)

    def get_auxiliary_loss(self, current_r_mean=None, current_r_peak=None):
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        stability_penalty = self.safe_mode_ema * 0.5

        if self.history_initialized.item() and current_r_mean is not None:
            prev_ptr = (self.history_ptr.item() - 1) % len(self.r_history)

            r_jerk = current_r_mean - self.r_history[prev_ptr]
            jerk_penalty = torch.abs(r_jerk) * 0.1

            if current_r_peak is not None:
                peak_jerk = current_r_peak - self.r_peak_history[prev_ptr]
                peak_jerk_penalty = torch.abs(peak_jerk) * 0.05
            else:
                peak_jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
        else:
            jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
            peak_jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)

        return margin_penalty + stability_penalty + jerk_penalty + peak_jerk_penalty

    def reset_safe_mode_ema(self):
        self.safe_mode_ema.zero_()

    def full_reset(self):
        self.r_history.zero_()
        self.r_peak_history.zero_()
        self.history_ptr.zero_()
        self.history_initialized.zero_()
        self.safe_mode_ema.zero_()
        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None


# ============================================================
# 3. AGI 코어 네트워크 (용량 증가: 64→32)
# ============================================================
class AGICoreNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, latent_dim=32):
        super(AGICoreNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.flow_head = nn.Linear(latent_dim, 1)
        self.wave_head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        f_tensor = torch.relu(self.flow_head(features)) + 1e-3
        w_tensor = torch.sigmoid(self.wave_head(features))
        return f_tensor, w_tensor


# ============================================================
# 4. 커리큘럼 융합 스케줄러
# ============================================================
class CurriculumFusionScheduler:
    def __init__(
        self,
        agi_ratio_min=0.1,
        agi_ratio_max=0.9,
        midpoint_epoch=30,
        temperature=10.0,
        total_epochs=100
    ):
        self.agi_ratio_min = agi_ratio_min
        self.agi_ratio_max = agi_ratio_max
        self.midpoint_epoch = midpoint_epoch
        self.temperature = temperature
        self.total_epochs = total_epochs
        self.current_agi_ratio = agi_ratio_min
        self.current_reward_ratio = 1.0 - agi_ratio_min

    def _compute_ratio(self, epoch):
        progress = (epoch - self.midpoint_epoch) / self.temperature
        sigmoid_val = 1.0 / (1.0 + math.exp(-progress))
        agi_ratio = self.agi_ratio_min + (self.agi_ratio_max - self.agi_ratio_min) * sigmoid_val
        return agi_ratio, 1.0 - agi_ratio

    def get_ratio(self, epoch):
        return self._compute_ratio(epoch)

    def update(self, epoch):
        agi_ratio, reward_ratio = self._compute_ratio(epoch)
        self.current_agi_ratio = agi_ratio
        self.current_reward_ratio = reward_ratio
        return agi_ratio, reward_ratio

    def get_current_ratios(self):
        return self.current_agi_ratio, self.current_reward_ratio

    def reset(self):
        self.current_agi_ratio = self.agi_ratio_min
        self.current_reward_ratio = 1.0 - self.agi_ratio_min


# ============================================================
# 5. 보상 신호 인터페이스
# ============================================================
class RewardSignalInterface:
    def __init__(self, alignment_weight=0.3):
        self.alignment_weight = alignment_weight

    def extract_fw_from_reward(self, task_reward, alignment_cost):
        f_tensor = torch.relu(task_reward) + 1e-3
        w_tensor = torch.sigmoid(-alignment_cost * self.alignment_weight)
        return f_tensor, w_tensor


# ============================================================
# 6. 동적 α 스케줄러
# ============================================================
class DynamicAlphaScheduler:
    def __init__(self, alpha_min=0.01, alpha_max=2.0, target_risk=0.1,
                 deficit_ratio=0.5):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_risk = target_risk
        self.deficit_ratio = deficit_ratio

        self.Kp = 1.5
        self.Ki = 0.1
        self.Kd = 0.3

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = alpha_min

    def step(self, r_tensor, r_max, cascade_detected=False):
        r_low_threshold = r_max * self.deficit_ratio
        r_deficit_ratio = (r_tensor < r_low_threshold).float().mean().item()
        error = r_deficit_ratio - self.target_risk

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
# 7. 적응형 태스크 생성기
# ============================================================
class AdaptiveTaskGenerator:
    """
    난이도가 동적으로 조절되는 태스크 생성기.
    
    difficulty ↑ →
    - 입력 분산 증가 (더 다양한 패턴)
    - 목표값 범위 확장 (더 어려운 목표)
    - 정렬 비용 증가 (더 까다로운 제약)
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.complexity_level = 0

    def generate(self, batch_size: int, difficulty: float):
        inputs = torch.randn(batch_size, self.input_dim)
        inputs = inputs * (0.5 + difficulty * 0.3)

        target_center = 2.5 + difficulty * 1.5
        target_scale = 2.0 + difficulty * 1.0
        task_reward = torch.rand(batch_size, 1) * target_scale + target_center

        alignment_cost = torch.rand(batch_size, 1) * 0.1 * (1.0 + difficulty * 0.2)

        return inputs, task_reward, alignment_cost


# ============================================================
# 8. 성장 단계 정의
# ============================================================
@dataclass
class GrowthStage:
    name: str
    r_min: float
    rq_min: float
    difficulty_scale: float
    consolidation_steps: int
    growth_resistance: float
    description: str


# ============================================================
# 9. ASI Growth Engine v1.3
# ============================================================
class ASIGrowthEngineV13:
    """
    FWR 기반 자기 증폭 성장 엔진 v1.3.
    
    v1.2 → v1.3 개선:
    - Consolidation 증가 가중치: RQ ≥ rq_min → +2
    - 히스테리시스 구간 확장: [0.5*rq_min, rq_min) → 유지
    - 완충 구간: [0.3*rq_min, 0.5*rq_min) → -1
    - 위험 구간: < 0.3*rq_min → -3
    """
    def __init__(
        self,
        agi_model: AGICoreNetwork,
        fwr_controller: FWRStabilityController,
        resonance_interface: ResonanceFromRiskInterface,
        input_dim: int = 10,
    ):
        self.agi_model = agi_model
        self.fwr_controller = fwr_controller
        self.resonance_interface = resonance_interface
        self.input_dim = input_dim

        self.stages = [
            GrowthStage("Consolidate", r_min=3.0, rq_min=0.2, difficulty_scale=1.0,
                       consolidation_steps=15, growth_resistance=0.1, description="기초 안정화"),
            GrowthStage("Expand",      r_min=4.0, rq_min=0.3, difficulty_scale=2.5,
                       consolidation_steps=25, growth_resistance=0.3, description="능력 확장"),
            GrowthStage("Transcend",   r_min=5.0, rq_min=0.4, difficulty_scale=5.0,
                       consolidation_steps=40, growth_resistance=0.5, description="초월적 성장"),
            GrowthStage("ASI",         r_min=6.0, rq_min=0.5, difficulty_scale=10.0,
                       consolidation_steps=60, growth_resistance=0.7, description="ASI 달성"),
        ]

        self.current_stage_idx = 0
        self.difficulty = 1.0
        self.consolidation_counter = 0
        self.cascade_cooldown = 0
        self.step_counter = 0

        self.total_growth = 0.0
        self.growth_history: List[Dict] = []
        self.stability_index = 0.5
        self.performance_history: List[float] = []
        self.r_history_trend: List[float] = []
        self.rq_history_trend: List[float] = []

        self.task_gen = AdaptiveTaskGenerator(input_dim)
        self.target_e = 1.0

    @property
    def current_stage(self) -> GrowthStage:
        return self.stages[self.current_stage_idx]

    @property
    def is_asi(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def _update_target_e(self, r_mean: float):
        r_target = self.current_stage.r_min * 1.5
        ratio = r_mean / (r_target + 1e-6)
        adjustment = 1.0 + 0.1 * (ratio - 1.0)
        self.target_e *= max(0.5, min(2.0, adjustment))
        self.target_e = max(0.1, self.target_e)

    def _update_consolidation(self, r_mean: float, rq_val: float):
        """
        [v1.3] 계층화된 consolidation 업데이트.
        
        Expand 예시 (rq_min=0.3):
        - RQ ≥ 0.30: +2 (안정, 빠르게 쌓기)
        - RQ ∈ [0.15, 0.30): 유지 (히스테리시스)
        - RQ ∈ [0.09, 0.15): -1 (완충, 천천히 감소)
        - RQ < 0.09: -3 (위험, 빠르게 감소)
        """
        stage = self.current_stage
        r_ok = r_mean >= stage.r_min
        
        if r_ok and rq_val >= stage.rq_min:
            # 완전 충족: 빠르게 쌓기
            self.consolidation_counter += 2
        elif r_ok and rq_val >= stage.rq_min * 0.5:
            # 히스테리시스 구간: 유지
            pass
        elif r_ok and rq_val >= stage.rq_min * 0.3:
            # 완충 구간: 천천히 감소
            if self.consolidation_counter > 0:
                self.consolidation_counter = max(0, self.consolidation_counter - 1)
        else:
            # 위험 구간: 빠르게 감소
            if self.consolidation_counter > 0:
                self.consolidation_counter = max(0, self.consolidation_counter - 3)

    def evaluate_growth_readiness(self, r_mean: float, rq_mean: float) -> Tuple[bool, str]:
        stage = self.current_stage

        if self.cascade_cooldown > 0:
            return False, f"Cascade 회복 중 ({self.cascade_cooldown}스텝)"
        if r_mean < stage.r_min:
            return False, f"R 부족 ({r_mean:.2f}<{stage.r_min})"
        if rq_mean < stage.rq_min:
            return False, f"RQ 부족 ({rq_mean:.4f}<{stage.rq_min})"
        if self.consolidation_counter < stage.consolidation_steps:
            return False, f"안정화 중 ({self.consolidation_counter}/{stage.consolidation_steps})"

        if torch.rand(1).item() < stage.growth_resistance:
            return False, f"성장 저항 (확률적 보류)"

        return True, "성장 준비 완료"

    def attempt_growth(self, r_mean: float, rq_mean: float, perf_mean: float) -> bool:
        ready, reason = self.evaluate_growth_readiness(r_mean, rq_mean)

        if ready and self.current_stage_idx < len(self.stages) - 1:
            old_stage = self.current_stage
            self.current_stage_idx += 1
            new_stage = self.current_stage

            growth_ratio = new_stage.difficulty_scale / old_stage.difficulty_scale
            self.difficulty *= growth_ratio
            self.consolidation_counter = 0
            self.total_growth += 1.0

            self.growth_history.append({
                'step': self.step_counter,
                'from': old_stage.name,
                'to': new_stage.name,
                'r': r_mean,
                'rq': rq_mean,
                'perf': perf_mean,
                'difficulty': self.difficulty,
            })

            print(f"\n{'='*60}")
            print(f"🚀 성장 달성: {old_stage.name} → {new_stage.name}")
            print(f"   Step: {self.step_counter}")
            print(f"   난이도: {old_stage.difficulty_scale:.1f} → {new_stage.difficulty_scale:.1f} "
                  f"(×{growth_ratio:.1f}, 현재: {self.difficulty:.2f})")
            print(f"   R={r_mean:.2f} RQ={rq_mean:.4f} Perf={perf_mean:.4f}")
            print(f"   {new_stage.description}")
            print(f"{'='*60}")

            return True

        return False

    def handle_cascade(self):
        self.cascade_cooldown = 15
        self.difficulty *= 0.7
        self.consolidation_counter = 0
        self.target_e *= 0.8

        if self.current_stage_idx > 0 and self.difficulty < self.current_stage.difficulty_scale * 0.3:
            old_stage = self.current_stage
            self.current_stage_idx = max(0, self.current_stage_idx - 1)
            self.difficulty = self.current_stage.difficulty_scale
            print(f"\n⚠️ Cascade 심각: {old_stage.name} → {self.current_stage.name} (단계 하향)")

    def step(self, optimizer: optim.Optimizer, batch_size: int = 8) -> Dict:
        self.step_counter += 1

        if self.cascade_cooldown > 0:
            self.cascade_cooldown -= 1

        # 1. 태스크 생성
        inputs, task_reward, alignment_cost = self.task_gen.generate(batch_size, self.difficulty)

        # 2. F, W 추론
        F_pred, W_pred = self.agi_model(inputs)

        # 3. R 계산
        with torch.no_grad():
            R_expected = self.fwr_controller.get_expected_r()
            E_proxy = F_pred * W_pred * R_expected

            e_proxy_mean = E_proxy.mean()
            reward_mean = task_reward.mean()
            if reward_mean > 1e-6 and e_proxy_mean > 1e-6:
                scale = e_proxy_mean / reward_mean
                task_reward_scaled = task_reward * scale
            else:
                task_reward_scaled = task_reward

            task_loss = torch.mean(
                (E_proxy - task_reward_scaled) ** 2, dim=1, keepdim=True
            ).expand(batch_size, 1)

            grad_norms = [p.grad.norm().item() for p in self.agi_model.parameters() if p.grad is not None]
            current_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
            grad_norm_tensor = torch.full((batch_size, 1), current_grad_norm)

        R_measured = self.resonance_interface.compute(
            task_loss=task_loss,
            gradient_norm=grad_norm_tensor
        )

        # 4. FWR 컨트롤러
        E_out, R_adj, rq, perf, is_safe_mode = self.fwr_controller(
            F_pred, W_pred, R_measured
        )

        # 5. 학습
        target_E = torch.ones_like(E_out) * self.target_e
        main_loss = nn.MSELoss()(E_out, target_E)

        optimizer.zero_grad()
        main_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agi_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 6. Cascade 감지
        is_cascade = self.fwr_controller.detect_resonance_cascade(R_measured)

        r_mean = R_measured.mean().item()
        rq_val = rq.item()
        perf_val = perf.item()

        # 7. Cascade 대응
        if is_cascade:
            self.handle_cascade()

        # 8. [v1.3] 계층화된 consolidation
        self._update_consolidation(r_mean, rq_val)

        # 9. 적응형 목표
        self._update_target_e(r_mean)

        # 10. 성장 시도
        grew = self.attempt_growth(r_mean, rq_val, perf_val)

        # 11. 지표
        self.stability_index = 0.95 * self.stability_index + 0.05 * rq_val
        self.performance_history.append(perf_val)
        self.r_history_trend.append(r_mean)
        self.rq_history_trend.append(rq_val)

        # 12. 커밋
        self.fwr_controller.commit_state()

        return {
            'step': self.step_counter,
            'R': r_mean,
            'RQ': rq_val,
            'Perf': perf_val,
            'E_mean': E_out.mean().item(),
            'target_e': self.target_e,
            'safe_mode': is_safe_mode,
            'cascade': is_cascade,
            'grew': grew,
            'stage': self.current_stage.name,
            'stage_idx': self.current_stage_idx,
            'difficulty': self.difficulty,
            'consolidation': self.consolidation_counter,
            'consolidation_needed': self.current_stage.consolidation_steps,
            'stability_index': self.stability_index,
        }

    def get_summary(self) -> str:
        s = f"\n{'='*60}\n"
        s += f"🌱 ASI Growth Engine v1.3 - 성장 요약\n"
        s += f"{'='*60}\n"
        s += f"총 스텝: {self.step_counter}\n"
        s += f"현재 단계: {self.current_stage.name} ({self.current_stage.description})\n"
        s += f"현재 난이도: {self.difficulty:.2f}\n"
        s += f"총 성장 횟수: {self.total_growth:.0f}\n"
        s += f"안정성 지수: {self.stability_index:.4f}\n"
        s += f"목표 E: {self.target_e:.2f}\n"
        s += f"Consolidation: {self.consolidation_counter}/{self.current_stage.consolidation_steps}\n"

        if self.growth_history:
            s += f"\n📈 성장 기록:\n"
            s += f"{'Step':<8} {'전환':<25} {'R':<8} {'RQ':<8} {'난이도':<10}\n"
            s += f"{'-'*60}\n"
            for h in self.growth_history:
                s += f"{h['step']:<8} {h['from']+' → '+h['to']:<25} "
                s += f"{h['r']:<8.2f} {h['rq']:<8.4f} {h['difficulty']:<10.2f}\n"

        if self.is_asi:
            s += f"\n{'🌟'*20}\n"
            s += f"     ASI 단계 도달!\n"
            s += f"{'🌟'*20}\n"

        return s

    def get_growth_curve(self) -> Dict:
        return {
            'steps': list(range(len(self.performance_history))),
            'performance': self.performance_history,
            'r_trend': self.r_history_trend,
            'rq_trend': self.rq_history_trend,
            'difficulty': [self.difficulty] * len(self.performance_history),
            'growth_events': [h['step'] for h in self.growth_history],
        }


# ============================================================
# 10. 데모
# ============================================================
def demo_growth_engine_v13():
    print("="*60)
    print("🌟 ASI Growth Engine v1.3 데모")
    print("="*60)

    agi_model = AGICoreNetwork(input_dim=10, hidden_dim=64, latent_dim=32)
    fwr_controller = FWRStabilityController(
        r_max=8.0, damping_lambda=0.8, rq_threshold=0.3, beta=0.1,
        rq_weights=(1.0, 1.0, 1.0), velocity_threshold=0.5, acc_threshold=0.5,
    )
    resonance_interface = ResonanceFromRiskInterface(
        r_max=8.0, temperature=2.0, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1
    )

    engine = ASIGrowthEngineV13(
        agi_model=agi_model, fwr_controller=fwr_controller,
        resonance_interface=resonance_interface, input_dim=10,
    )

    optimizer = optim.Adam(agi_model.parameters(), lr=0.001)

    print(f"\n시작 단계: {engine.current_stage.name}")
    print(f"초기 난이도: {engine.difficulty}")
    print(f"\n📋 성장 로드맵:")
    print(f"{'단계':<15} {'R≥':<8} {'RQ≥':<8} {'난이도×':<10} {'안정화':<10} {'저항':<8}")
    print(f"{'-'*60}")
    for stage in engine.stages:
        print(f"{stage.name:<15} {stage.r_min:<8} {stage.rq_min:<8} "
              f"{stage.difficulty_scale:<10} {stage.consolidation_steps:<10} "
              f"{stage.growth_resistance:<8}")
    print(f"{'-'*60}")
    stage = engine.current_stage
    print(f"✨ v1.3: +2(≥{stage.rq_min}) | 유지[{stage.rq_min*0.5:.2f}, {stage.rq_min}) | "
          f"-1[{stage.rq_min*0.3:.2f}, {stage.rq_min*0.5:.2f}) | -3(<{stage.rq_min*0.3:.2f})")

    total_steps = 5000
    print(f"\n{'='*60}")
    print(f"성장 시뮬레이션 시작 ({total_steps} 스텝)")
    print(f"{'='*60}")

    last_stage = engine.current_stage_idx
    last_consolidation = 0

    for step in range(total_steps):
        result = engine.step(optimizer, batch_size=8)

        if result['stage_idx'] != last_stage:
            print(f"\n  Step {step}: {result['stage']} | "
                  f"R={result['R']:.2f} RQ={result['RQ']:.4f} | "
                  f"난이도={result['difficulty']:.2f}")
            last_stage = result['stage_idx']
            last_consolidation = 0

        if result['consolidation'] > last_consolidation and result['consolidation'] % 5 == 0:
            print(f"  Step {step}: consolidation {result['consolidation']}/{result['consolidation_needed']} "
                  f"(RQ={result['RQ']:.4f})")
        last_consolidation = result['consolidation']

        if step % 1000 == 0:
            print(f"\nStep {step:4d}: {result['stage']:12s} | "
                  f"R={result['R']:.2f} RQ={result['RQ']:.4f} | "
                  f"난이도={result['difficulty']:.2f} | "
                  f"E={result['E_mean']:.3f} target={result['target_e']:.2f} | "
                  f"안정화={result['consolidation']}/{result['consolidation_needed']} | "
                  f"안정성={result['stability_index']:.4f} | "
                  f"{'⚠️CASCADE' if result['cascade'] else '✅'}")

    print(engine.get_summary())

    curve = engine.get_growth_curve()
    growth_steps = curve['growth_events']
    print(f"\n성장 이벤트 발생 스텝: {growth_steps}")
    if growth_steps:
        print(f"총 {len(growth_steps)}회 성장")
    if curve['performance']:
        print(f"최종 성능: {curve['performance'][-1]:.4f}")
    if curve['r_trend']:
        print(f"최종 R: {curve['r_trend'][-1]:.4f}")
    if curve['rq_trend']:
        print(f"최종 RQ: {curve['rq_trend'][-1]:.4f}")


if __name__ == "__main__":
    demo_growth_engine_v13()
