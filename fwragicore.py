"""
FWR Stability Controller v3.7.1
- detect_resonance_cascade: 실시간 텐서 반영 + peak 조건 추가
- get_auxiliary_loss: current_r_mean/peak 인자로 gradient 연결
- 학습 루프 호출부 수정
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
        self.confidences = torch.sigmoid(torch.randn(n_samples, 1) * 0.5 + 0.5)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.task_rewards[idx],
            self.alignment_costs[idx],
            self.confidences[idx]
        )


# ============================================================
# 1. FWR 안정성 제어기 v3.7.1
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

    def forward(self, f_tensor, w_tensor, r_tensor):
        # 1. 적응형 감쇠
        r_excess = F_pt.relu(r_tensor - self.r_max)

        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor

        # 2. 창발 에너지 (performance)
        e_tensor = f_tensor * w_tensor * r_adj

        # 3. resonance_quality: pure stability metric
        r_std = torch.std(r_tensor) + 1e-8
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
        [v3.7.1] 실시간 텐서를 직접 반영하여 링 버퍼 지연 없는 폭주 감지.
        peak 조건 추가: 배치 내 단일 아웃라이어 폭주도 감지.

        cascade 조건:
            (mean > r_max*1.2 OR peak > r_max*1.5)
            AND velocity > v_threshold
            AND acceleration > a_threshold
        """
        if not self.history_initialized.item():
            return False

        ptr = self.history_ptr.item()

        idx1 = (ptr - 1) % len(self.r_history)
        idx2 = (ptr - 2) % len(self.r_history)

        r1 = self.r_history[idx1]
        r2 = self.r_history[idx2]

        # 실시간 텐서가 있으면 현재 상태로 사용 (링 버퍼 지연 제거)
        if current_r_tensor is not None:
            r0 = current_r_tensor.detach().mean()
            p0 = current_r_tensor.detach().max()
        else:
            r0 = r1
            p0 = self.r_peak_history[idx1]

        v1 = r0 - r1
        v2 = r1 - r2
        acc = v1 - v2

        # mean 폭주 OR peak 폭주 둘 다 감지
        above_threshold = (r0 > self.r_max * 1.2) or (p0 > self.r_max * 1.5)
        is_rising = v1 > self.velocity_threshold
        is_accelerating = acc > self.acc_threshold

        return bool(above_threshold and is_rising and is_accelerating)

    def get_auxiliary_loss(self, current_r_mean=None, current_r_peak=None):
        """
        [v3.7.1] current_r_mean/peak를 인자로 받아 gradient 연결 복구.
        과거 버퍼(detach)와 현재 예측값(grad 연결) 간 jerk 계산.
        """
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        stability_penalty = self.safe_mode_ema * 0.5

        if self.history_initialized.item() and current_r_mean is not None:
            prev_ptr = (self.history_ptr.item() - 1) % len(self.r_history)

            # 현재 배치 예측값 ↔ 직전 배치 버퍼값: gradient 정상 흐름
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
# 3. 커리큘럼 융합 스케줄러
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
# 5. 동적 α 스케줄러
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
# 6. 유틸리티
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
    print("Shape 검증")
    print("=" * 70)

    batch = next(iter(dataloader))
    inputs, task_reward, alignment_cost, confidence = batch
    print(f"inputs: {inputs.shape} | task_reward: {task_reward.shape} | "
          f"alignment_cost: {alignment_cost.shape} | confidence: {confidence.shape}")

    with torch.no_grad():
        F, W, R = agi_model(inputs)
        print(f"F: {F.shape} | W: {W.shape} | R: {R.shape}")

    assert inputs.dim() == 2
    assert inputs.shape[-1] == 10
    assert F.shape == (dataloader.batch_size, 1)
    assert W.shape == (dataloader.batch_size, 1)
    assert R.shape == (dataloader.batch_size, 1)
    print("✅ 모든 shape 검증 통과")


def demo_resonance_quality():
    """F/W 변화가 resonance_quality에 영향 없음을 확인"""
    print("\n" + "=" * 70)
    print("resonance_quality vs performance_score 분리 검증 (v3.7.1)")
    print("=" * 70)

    controller = FWRStabilityController(r_max=8.0, beta=0.1)

    test_cases = [
        ("F=1,   W=1, R=4 균일",    1.0,   1.0, [4.0,  4.0,  4.0,  4.0 ]),
        ("F=10,  W=1, R=4 균일",    10.0,  1.0, [4.0,  4.0,  4.0,  4.0 ]),
        ("F=100, W=1, R=4 균일",    100.0, 1.0, [4.0,  4.0,  4.0,  4.0 ]),
        ("F=1,   W=1, R=0 dead",    1.0,   1.0, [0.01, 0.01, 0.01, 0.01]),
        ("F=1,   W=1, R=20 과잉",   1.0,   1.0, [20.0, 20.0, 20.0, 20.0]),
        ("F=1,   W=1, R 혼합 1~20", 1.0,   1.0, [1.0,  5.0,  10.0, 20.0]),
    ]

    print(f"{'케이스':<28} {'RQ':<8} {'Perf':<10} {'안전모드'}")
    print("-" * 60)

    for name, f_val, w_val, r_vals in test_cases:
        F = torch.tensor([[f_val]] * 4)
        W = torch.tensor([[w_val]] * 4)
        R = torch.tensor([[v] for v in r_vals])
        _, _, rq, perf, safe = controller(F, W, R)
        controller.commit_state()
        print(f"{name:<28} {rq.item():<8.4f} {perf.item():<10.4f} "
              f"{'⚠️YES' if safe else '✅NO'}")

    print("-" * 60)
    print("기대: F/W 변화 → RQ 불변, Perf만 변함")


def demo_cascade_detection():
    """[v3.7.1] 실시간 텐서 + peak 조건 cascade 감지 검증"""
    print("\n" + "=" * 70)
    print("Cascade 감지 검증 (v3.7.1): 실시간 텐서 반영 + peak 조건")
    print("=" * 70)

    controller = FWRStabilityController(
        r_max=8.0, velocity_threshold=0.5, acc_threshold=0.5
    )
    F = torch.tensor([[1.0]])
    W = torch.tensor([[1.0]])

    scenarios = {
        "고점 유지 (plateau)":          [10.0, 10.0, 10.0, 10.5],
        "noise (8→8.1→8.15)":           [7.0,  8.0,  8.1,  8.15],
        "완만한 상승":                   [7.0,  8.5,  9.0,  9.6 ],
        "폭주 trajectory":               [5.0,  10.0, 15.0, 25.0],
        "감소 중":                       [20.0, 15.0, 10.0, 8.0 ],
        "peak spike (mean은 정상)":      [2.0,  2.0,  2.0,  2.0 ],  # peak만 높은 케이스
    }

    # peak spike 케이스: 배치 내 한 원소만 극단값
    peak_spike_r = torch.tensor([[2.0], [2.0], [2.0], [15.0]])

    print(f"{'시나리오':<30} {'Cascade'}")
    print("-" * 44)

    for name, r_seq in scenarios.items():
        controller.full_reset()

        if name == "peak spike (mean은 정상)":
            # 히스토리 3개 쌓기
            for r_val in [5.0, 10.0, 12.0]:
                R = torch.tensor([[r_val]])
                controller(F, W, R)
                controller.commit_state()
            # 현재 배치: mean=5.25이지만 peak=15
            _, _, _, _, _ = controller(F, W, torch.tensor([[2.0]]))
            result = controller.detect_resonance_cascade(peak_spike_r)
        else:
            for r_val in r_seq:
                R = torch.tensor([[r_val]])
                controller(F, W, R)
                controller.commit_state()
            result = controller.detect_resonance_cascade(
                torch.tensor([[r_seq[-1]]])
            )

        print(f"{name:<30} {'⚠️ YES' if result else '✅ NO'}")

    print("-" * 44)
    print("기대: 고점유지/noise→NO | 폭주/peak spike→YES | 감소→NO")


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
    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()),
        lr=0.001
    )
    criterion_main = nn.MSELoss()

    ema_reset_interval = 50
    integral_reset_interval = 20

    print("\n" + "=" * 70)
    print("FWR AGI Training with Curriculum Fusion (v3.7.1)")
    print(f"Dataset: {len(dataloader.dataset)} samples | Batch: {dataloader.batch_size}")
    print(f"Fusion: agi=[{fusion_scheduler.agi_ratio_min:.1f}→{fusion_scheduler.agi_ratio_max:.1f}] "
          f"mid={fusion_scheduler.midpoint_epoch} temp={fusion_scheduler.temperature}")
    print(f"PID α: [{alpha_scheduler.alpha_min}, {alpha_scheduler.alpha_max}] "
          f"target_risk={alpha_scheduler.target_risk}")
    print(f"rq_threshold={fwr_controller.rq_threshold} | beta={fwr_controller.beta} | "
          f"v_th={fwr_controller.velocity_threshold} | a_th={fwr_controller.acc_threshold}")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_main_loss = 0.0
        epoch_total_loss = 0.0
        epoch_rq = 0.0
        epoch_perf = 0.0
        n_batches = 0
        last_cascade = False

        agi_ratio, reward_ratio = fusion_scheduler.update(epoch)

        for batch in dataloader:
            inputs, task_reward, alignment_cost, confidence = batch

            F_pred, W_pred, R_pred = agi_model(inputs)

            F_reward, W_reward, R_reward = reward_interface.extract_fwr_from_reward(
                task_reward, alignment_cost, confidence
            )

            F_combined = F_pred * agi_ratio + F_reward * reward_ratio
            W_combined = W_pred * agi_ratio + W_reward * reward_ratio
            R_combined = R_pred * agi_ratio + R_reward * reward_ratio

            E_out, R_adj, rq, perf, is_safe_mode = fwr_controller(
                F_combined, W_combined, R_combined
            )

            # [v3.7.1] 실시간 텐서 직접 전달 (링 버퍼 지연 제거)
            is_cascade = fwr_controller.detect_resonance_cascade(R_combined)
            last_cascade = is_cascade

            dynamic_alpha = alpha_scheduler.step(R_combined, fwr_controller.r_max, is_cascade)

            target_E = torch.ones_like(E_out) * target_e_value
            main_loss = criterion_main(E_out, target_E)

            # [v3.7.1] current_r_mean/peak 전달로 gradient 연결
            aux_loss = fwr_controller.get_auxiliary_loss(
                current_r_mean=R_combined.mean(),
                current_r_peak=R_combined.max()
            )
            total_loss = main_loss + dynamic_alpha * aux_loss

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
            print(f"Epoch {epoch+1:03d} | Total: {avg_total:.4f} Main: {avg_main:.4f} | "
                  f"RQ: {avg_rq:.4f} Perf: {avg_perf:.4f} | "
                  f"α: {alpha_scheduler.current_alpha:.4f} | "
                  f"AGI%: {agi_ratio:.2f} | "
                  f"SafeEMA: {fwr_controller.safe_mode_ema.item():.4f} | "
                  f"Cascade: {'⚠️' if last_cascade else '✅'}")

    return agi_model, fwr_controller, alpha_scheduler


# ============================================================
# 8. 추론 테스트
# ============================================================
def test_trained_model(agi_model, fwr_controller, fusion_scheduler):
    print("\n" + "=" * 70)
    print("추론 테스트")
    print("=" * 70)

    agi_model.eval()
    fwr_controller.eval()

    with torch.no_grad():
        inputs_normal = torch.randn(4, 10)
        F, W, R = agi_model(inputs_normal)
        E_out, R_adj, rq, perf, safe = fwr_controller(F, W, R)

        print(f"F:\n{F}")
        print(f"W:\n{W}")
        print(f"R (원본):\n{R}")
        print(f"R (조정):\n{R_adj}")
        print(f"E:\n{E_out}")
        print(f"resonance_quality: {rq.item():.4f}")
        print(f"performance_score: {perf.item():.4f}")
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
    alpha_scheduler = DynamicAlphaScheduler(alpha_min=0.01, alpha_max=2.0, target_risk=0.1)
    reward_interface = RewardSignalInterface(alignment_weight=0.3, confidence_scale=5.0)

    fusion_scheduler = CurriculumFusionScheduler(
        agi_ratio_min=0.1,
        agi_ratio_max=0.9,
        midpoint_epoch=20,
        temperature=8.0,
        total_epochs=EPOCHS
    )

    print_fusion_schedule(fusion_scheduler, epochs=EPOCHS, interval=10)
    validate_tensor_shapes(agi_model, dataloader)
    demo_resonance_quality()
    demo_cascade_detection()

    agi_model, fwr_controller, alpha_scheduler = train_fwr_agi_with_curriculum(
        agi_model, fwr_controller, alpha_scheduler, reward_interface,
        fusion_scheduler, dataloader, epochs=EPOCHS, target_e_value=5.0
    )

    print(f"\n{'='*70}")
    print("✅ 학습 완료")
    print(f"  safety_margin:  {fwr_controller.safety_margin.item():.4f}")
    print(f"  최종 α:         {alpha_scheduler.current_alpha:.4f}")
    print(f"  safe_mode_ema:  {fwr_controller.safe_mode_ema.item():.4f}")
    print(f"  최종 융합 비율: AGI={fusion_scheduler.current_agi_ratio:.4f}, "
          f"Reward={fusion_scheduler.current_reward_ratio:.4f}")

    test_trained_model(agi_model, fwr_controller, fusion_scheduler)
