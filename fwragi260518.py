
"""
FWR Stability Controller v3.8.5
- get_expected_r(): 하한선 r_max * 0.3 적용 (R 기대값 최소 보장)
- RQ-SafeEMA 균형 복원
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

    위험 구성:
        risk = α·task_loss + β·entropy + γ·grad_excess + δ·weight_excess
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
        """
        [v3.8.5] 히스토리 기반 기대 R값.
        - 초기화 상태: r_max * 0.5
        - 일반: max(히스토리 평균, r_max * 0.3)  ← 하한선 적용
        """
        if not self.history_initialized.item():
            return self.r_max * 0.5
        
        ptr = self.history_ptr.item()
        if ptr == 0:
            return self.r_max * 0.5
        
        mean_r = self.r_history[:ptr].mean().item()
        # 하한선: r_max * 0.3 (절대 이 아래로 내려가지 않음)
        return max(mean_r, self.r_max * 0.3)

    def forward(self, f_tensor, w_tensor, r_tensor):
        # 1. 적응형 감쇠
        r_excess = F_pt.relu(r_tensor - self.r_max)

        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor

        # 2. 창발 에너지: E = F × W × R
        e_tensor = f_tensor * w_tensor * r_adj

        # 3. resonance_quality
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

        # 4. performance_score
        performance_score = torch.mean(e_tensor)

        # 5. 점진적 안전 모드
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
        """
        공명 붕괴 + 과잉 폭주 감지.
        붕괴: R < r_max*0.2 AND 하락 속도 초과 (velocity 단독)
        폭주: R > r_max*1.2 AND 상승 속도+가속도 초과
        """
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

        # 공명 붕괴: velocity 단독
        collapse = (
            (r0 < self.r_max * 0.2) and
            (v1 < -self.velocity_threshold)
        )

        # 과잉 폭주: velocity + acceleration
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
# 3. AGI 코어 네트워크
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
# 7. 유틸리티
# ============================================================
def print_fusion_schedule(scheduler, epochs=100, interval=10):
    print("\n" + "=" * 70)
    print("커리큘럼 융합 비율 스케줄 (Stateless 미리보기)")
    print("=" * 70)
    print(f"{'Epoch':<10} {'AGI Ratio':<12} {'Reward Ratio':<12} {'설명'}")
    print("-" * 60)

    for ep in range(0, epochs + 1, interval):
        agi_r, rew_r = scheduler.get_ratio(ep)
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
    curr_agi, _ = scheduler.get_current_ratios()
    status = '✅ 정상' if abs(curr_agi - scheduler.agi_ratio_min) < 1e-6 else '❌ 오염됨'
    print(f"현재 저장된 상태: AGI={curr_agi:.4f} ({status})")


def validate_tensor_shapes(agi_model, dataloader):
    print("\n" + "=" * 70)
    print("Shape 검증 (v3.8.5)")
    print("=" * 70)

    batch = next(iter(dataloader))
    inputs, task_reward, alignment_cost = batch
    print(f"inputs: {inputs.shape} | task_reward: {task_reward.shape} | "
          f"alignment_cost: {alignment_cost.shape}")

    with torch.no_grad():
        F, W = agi_model(inputs)
        print(f"F: {F.shape} | W: {W.shape}")

    assert inputs.dim() == 2
    assert inputs.shape[-1] == 10
    assert F.shape == (dataloader.batch_size, 1)
    assert W.shape == (dataloader.batch_size, 1)
    print("✅ 모든 shape 검증 통과")


def demo_resonance_conversion():
    print("\n" + "=" * 70)
    print("ResonanceFromRiskInterface: 위험 → 공명 변환")
    print("r_max=8.0, temperature=2.0")
    print("=" * 70)

    interface = ResonanceFromRiskInterface(r_max=8.0, temperature=2.0)

    scenarios = [
        ("정상 학습 (loss=0.1)",          0.1,  0.2,  0.5,  2.0),
        ("약한 불안정 (loss=1.0)",         1.0,  0.5,  0.8,  3.0),
        ("불안정 (loss=3.0)",             3.0,  1.5,  1.2,  4.0),
        ("혼돈 (loss=5.0, ent=3.0)",      5.0,  3.0,  2.0,  6.0),
        ("폭주 (loss=10, grad=5.0)",     10.0,  2.0,  5.0, 15.0),
    ]

    print(f"{'시나리오':<35} {'R (공명)':<10} {'의미'}")
    print("-" * 60)

    for name, loss, ent, grad_norm, w_norm in scenarios:
        r = interface.compute(
            task_loss=torch.tensor([[loss]]),
            prediction_entropy=torch.tensor([[ent]]),
            gradient_norm=torch.tensor([[grad_norm]]),
            weight_norm=torch.tensor([[w_norm]])
        )
        if r.item() > 6.0:
            desc = "🟢 강한 공명 (안정)"
        elif r.item() > 3.0:
            desc = "🟡 중간 공명"
        elif r.item() > 1.0:
            desc = "🟠 약한 공명 (주의)"
        else:
            desc = "🔴 공명 붕괴 (위험)"
        print(f"{name:<35} {r.item():<10.4f} {desc}")

    print("-" * 60)


def demo_fwr_formula():
    print("\n" + "=" * 70)
    print("E = F × W × R 수식 검증")
    print("=" * 70)

    controller = FWRStabilityController(r_max=8.0, beta=0.1)
    interface = ResonanceFromRiskInterface(r_max=8.0, temperature=2.0)

    F = torch.tensor([[5.0]] * 4)
    W = torch.tensor([[0.8]] * 4)

    scenarios = [
        ("정상 학습",  0.1,  0.2),
        ("불안정",     3.0,  1.5),
        ("폭주",      10.0,  3.0),
    ]

    print(f"{'시나리오':<15} {'R':<8} {'E':<10} {'RQ':<8} {'안전모드'}")
    print("-" * 55)

    for name, loss, ent in scenarios:
        R = interface.compute(
            task_loss=torch.tensor([[loss]] * 4),
            prediction_entropy=torch.tensor([[ent]] * 4)
        )
        E_out, _, rq, perf, safe = controller(F, W, R)
        controller.commit_state()
        print(f"{name:<15} {R.mean().item():<8.4f} {E_out.mean().item():<10.4f} "
              f"{rq.item():<8.4f} {'⚠️YES' if safe else '✅NO'}")

    print("-" * 55)


def demo_cascade_detection():
    print("\n" + "=" * 70)
    print("Cascade 감지: 붕괴=velocity 단독 | 폭주=velocity+acc")
    print("=" * 70)

    controller = FWRStabilityController(
        r_max=8.0, velocity_threshold=0.5, acc_threshold=0.5
    )
    F = torch.tensor([[1.0]])
    W = torch.tensor([[1.0]])

    scenarios = {
        "정상 유지":               [5.0,  5.0,  5.0,  5.0 ],
        "완만한 하락":             [5.0,  4.0,  3.5,  3.0 ],
        "공명 붕괴 trajectory":    [5.0,  3.0,  1.5,  0.3 ],
        "과잉 폭주 trajectory":    [2.0,  5.0,  12.0, 22.0],
        "회복 중 (상승)":          [1.0,  2.0,  4.0,  6.0 ],
    }

    print(f"{'시나리오':<28} {'Cascade'}")
    print("-" * 42)

    for name, r_seq in scenarios.items():
        controller.full_reset()
        for r_val in r_seq[:-1]:
            R = torch.tensor([[r_val]])
            controller(F, W, R)
            controller.commit_state()
        current_r = torch.tensor([[r_seq[-1]]])
        result = controller.detect_resonance_cascade(current_r)
        print(f"{name:<28} {'⚠️ YES' if result else '✅ NO'}")

    print("-" * 42)


# ============================================================
# 8. 통합 학습 루프 v3.8.5
# ============================================================
def train_fwr_agi_v385(
    agi_model,
    fwr_controller,
    alpha_scheduler,
    reward_interface,
    resonance_interface,
    fusion_scheduler,
    dataloader,
    epochs=100,
    target_e_value=None
):
    """
    [v3.8.5] get_expected_r() 하한선 r_max*0.3 적용.
    R이 일시적으로 낮아져도 기대 R이 최소 2.4를 유지 → loss 과대평가 방지.
    """
    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()),
        lr=0.001
    )
    criterion_task = nn.MSELoss()

    ema_reset_interval = 50
    integral_reset_interval = 20
    auto_target = target_e_value is None

    print("\n" + "=" * 70)
    print("FWR AGI Training v3.8.5 (R_expected 하한선 r_max*0.3)")
    print(f"Dataset: {len(dataloader.dataset)} samples | Batch: {dataloader.batch_size}")
    print(f"target_e_value: {'auto' if auto_target else target_e_value}")
    print(f"rq_threshold={fwr_controller.rq_threshold} | r_max={fwr_controller.r_max}")
    print(f"R_expected 하한선: {fwr_controller.r_max * 0.3:.1f}")
    print(f"deficit_ratio={alpha_scheduler.deficit_ratio}")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_main_loss = 0.0
        epoch_total_loss = 0.0
        epoch_rq = 0.0
        epoch_perf = 0.0
        epoch_r_mean = 0.0
        epoch_r_expected = 0.0
        n_batches = 0
        last_cascade = False

        agi_ratio, reward_ratio = fusion_scheduler.update(epoch)

        for batch in dataloader:
            inputs, task_reward, alignment_cost = batch

            # 1. AGI 코어: F, W
            F_pred, W_pred = agi_model(inputs)

            # 2. 보상 신호 → F, W
            F_reward, W_reward = reward_interface.extract_fw_from_reward(
                task_reward, alignment_cost
            )

            # 3. 커리큘럼 융합
            F_combined = F_pred * agi_ratio + F_reward * reward_ratio
            W_combined = W_pred * agi_ratio + W_reward * reward_ratio

            # 4. [v3.8.5] R 기대값 기반 task_loss_proxy
            with torch.no_grad():
                R_expected = fwr_controller.get_expected_r()  # 하한선 r_max*0.3 적용
                
                # FWR 컨텍스트 proxy E
                E_proxy = F_combined * W_combined * R_expected
                
                # 스케일 정규화
                e_proxy_mean = E_proxy.mean()
                reward_mean = task_reward.mean()
                if reward_mean > 1e-6 and e_proxy_mean > 1e-6:
                    scale = e_proxy_mean / reward_mean
                    task_reward_scaled = task_reward * scale
                else:
                    task_reward_scaled = task_reward
                
                task_loss_proxy = torch.mean(
                    (E_proxy - task_reward_scaled) ** 2, dim=1, keepdim=True
                )
                task_loss_proxy = task_loss_proxy.expand(inputs.shape[0], 1)

                # grad_norm
                grad_norms = [p.grad.norm().item()
                              for p in agi_model.parameters()
                              if p.grad is not None]
                current_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
                grad_norm_tensor = torch.full(
                    (inputs.shape[0], 1), current_grad_norm,
                    device=inputs.device
                )

            # 5. R 계산
            R_measured = resonance_interface.compute(
                task_loss=task_loss_proxy,
                gradient_norm=grad_norm_tensor
            )

            # 6. FWR 컨트롤러
            E_out, R_adj, rq, perf, is_safe_mode = fwr_controller(
                F_combined, W_combined, R_measured
            )

            # 7. target_e_value 자동 설정
            if auto_target and target_e_value is None and epoch == 0 and n_batches == 0:
                target_e_value = perf.item()
                print(f"  [Auto target_e_value = {target_e_value:.4f}]")

            # 8. 폭주 감지
            is_cascade = fwr_controller.detect_resonance_cascade(R_measured)
            last_cascade = is_cascade

            # 9. 동적 α
            dynamic_alpha = alpha_scheduler.step(
                R_measured, fwr_controller.r_max, is_cascade
            )

            # 10. Loss
            target_E = torch.ones_like(E_out) * target_e_value
            main_loss = criterion_task(E_out, target_E)
            aux_loss = fwr_controller.get_auxiliary_loss(
                current_r_mean=R_measured.mean(),
                current_r_peak=R_measured.max()
            )
            total_loss = main_loss + dynamic_alpha * aux_loss

            # 11. 역전파
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agi_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(fwr_controller.parameters(), max_norm=1.0)
            optimizer.step()

            fwr_controller.commit_state()

            epoch_main_loss += main_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_rq += rq.item()
            epoch_perf += perf.item()
            epoch_r_mean += R_measured.mean().item()
            epoch_r_expected += R_expected
            n_batches += 1

        if (epoch + 1) % ema_reset_interval == 0:
            fwr_controller.reset_safe_mode_ema()

        if (epoch + 1) % integral_reset_interval == 0:
            alpha_scheduler.reset_integral()

        if (epoch + 1) % 10 == 0:
            avg_main = epoch_main_loss / max(n_batches, 1)
            avg_total = epoch_total_loss / max(n_batches, 1)
            avg_rq = epoch_rq / max(n_batches, 1)
            avg_perf = epoch_perf / max(n_batches, 1)
            avg_r = epoch_r_mean / max(n_batches, 1)
            avg_r_exp = epoch_r_expected / max(n_batches, 1)
            print(f"Epoch {epoch+1:03d} | Total: {avg_total:.4f} Main: {avg_main:.4f} | "
                  f"RQ: {avg_rq:.4f} Perf: {avg_perf:.4f} R: {avg_r:.4f} R_exp: {avg_r_exp:.1f} | "
                  f"α: {alpha_scheduler.current_alpha:.4f} | "
                  f"AGI%: {agi_ratio:.2f} | "
                  f"SafeEMA: {fwr_controller.safe_mode_ema.item():.4f} | "
                  f"Cascade: {'⚠️' if last_cascade else '✅'}")

    return agi_model, fwr_controller, alpha_scheduler


# ============================================================
# 9. 추론 테스트
# ============================================================
def test_trained_model(agi_model, fwr_controller, resonance_interface, fusion_scheduler):
    print("\n" + "=" * 70)
    print("추론 테스트 (v3.8.5)")
    print("=" * 70)

    agi_model.eval()
    fwr_controller.eval()

    with torch.no_grad():
        inputs_normal = torch.randn(4, 10)
        F, W = agi_model(inputs_normal)

        R = resonance_interface.compute(
            task_loss=torch.ones(4, 1) * 0.1,
            gradient_norm=torch.ones(4, 1) * 0.3
        )

        E_out, R_adj, rq, perf, safe = fwr_controller(F, W, R)

        print(f"F (추진력):\n{F}")
        print(f"W (구조력):\n{W}")
        print(f"R (공명): mean={R.mean().item():.4f}")
        print(f"R (조정): mean={R_adj.mean().item():.4f}")
        print(f"E (창발 에너지):\n{E_out}")
        print(f"resonance_quality: {rq.item():.4f}")
        print(f"performance_score: {perf.item():.4f}")
        print(f"R_expected: {fwr_controller.get_expected_r():.1f}")
        print(f"안전 모드: {'⚠️ YES' if safe else '✅ NO'}")
        print(f"폭주 감지: {'⚠️ YES' if fwr_controller.detect_resonance_cascade(R) else '✅ NO'}")
        agi_r, rew_r = fusion_scheduler.get_current_ratios()
        print(f"융합 비율: AGI={agi_r:.4f}, Reward={rew_r:.4f}")

    agi_model.train()
    fwr_controller.train()


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    BATCH_SIZE = 8
    N_SAMPLES = 1000
    INPUT_DIM = 10
    EPOCHS = 60

    dataset = AGIAlignmentDataset(n_samples=N_SAMPLES, input_dim=INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    agi_model = AGICoreNetwork(input_dim=INPUT_DIM, hidden_dim=32, latent_dim=16)
    fwr_controller = FWRStabilityController(
        r_max=8.0,
        damping_lambda=0.8,
        rq_threshold=0.3,
        beta=0.1,
        rq_weights=(1.0, 1.0, 1.0),
        velocity_threshold=0.5,
        acc_threshold=0.5,
    )
    alpha_scheduler = DynamicAlphaScheduler(
        alpha_min=0.01,
        alpha_max=2.0,
        target_risk=0.1,
        deficit_ratio=0.5
    )
    reward_interface = RewardSignalInterface(alignment_weight=0.3)
    resonance_interface = ResonanceFromRiskInterface(
        r_max=8.0,
        temperature=2.0,
        alpha=0.4, beta=0.3, gamma=0.2, delta=0.1
    )
    fusion_scheduler = CurriculumFusionScheduler(
        agi_ratio_min=0.1,
        agi_ratio_max=0.9,
        midpoint_epoch=20,
        temperature=8.0,
        total_epochs=EPOCHS
    )

    print_fusion_schedule(fusion_scheduler, epochs=EPOCHS, interval=10)
    validate_tensor_shapes(agi_model, dataloader)
    demo_resonance_conversion()
    demo_fwr_formula()
    demo_cascade_detection()

    agi_model, fwr_controller, alpha_scheduler = train_fwr_agi_v385(
        agi_model, fwr_controller, alpha_scheduler, reward_interface,
        resonance_interface, fusion_scheduler, dataloader,
        epochs=EPOCHS, target_e_value=None
    )

    print(f"\n{'='*70}")
    print("✅ 학습 완료 (v3.8.5)")
    print(f"  safety_margin:  {fwr_controller.safety_margin.item():.4f}")
    print(f"  최종 α:         {alpha_scheduler.current_alpha:.4f}")
    print(f"  safe_mode_ema:  {fwr_controller.safe_mode_ema.item():.4f}")
    print(f"  R_expected:     {fwr_controller.get_expected_r():.1f}")
    print(f"  최종 융합 비율: AGI={fusion_scheduler.current_agi_ratio:.4f}, "
          f"Reward={fusion_scheduler.current_reward_ratio:.4f}")

    test_trained_model(agi_model, fwr_controller, resonance_interface, fusion_scheduler)
