import torch
import math
import logging
from typing import Tuple, Callable, Optional

logger = logging.getLogger("FWR")
logger.setLevel(logging.INFO)


class FWRStabilityController:
    """
    FWR Stability Controller — v2.1 (Production + FWR v3.1 연계)

    v2.0 대비 수정:
    1. reset()을 soft/hard로 분리 — T는 Safe Mode 해제 시 보존
    2. T 누적을 단조증가 보장 (EMA 대신 누적합 + decay 옵션)
    3. project_with_adaptive_step에서 no_grad 데코레이터 제거,
       내부에서 명시적으로 grad context 관리
    """

    def __init__(
        self,
        lambda_max_estimator: Callable[[Optional[torch.Tensor], torch.Tensor], float],
        potential_function: Callable[[torch.Tensor], torch.Tensor],
        *,
        theta_enter: float = 0.38,
        theta_exit: float = 0.52,
        alpha: float = 12.0,
        max_tightening: float = 5.0,
        recovery_steps: int = 100,
        proj_lr: float = 4e-3,
        proj_steps_max: int = 10,
        proj_steps_min: int = 3,
        grad_clip_max_norm: float = 4.0,
        cooldown_k: float = 3.5,
        expl_decay_k: float = 4.5,
        phi_improve_tol: float = 1e-5,
        lambda_clip_margin: float = 5.0,
        # FWR v3.1 연계 파라미터
        resonance_decay: float = 0.92,
        t_lambda: float = 0.0,   # T decay 상수 (0이면 완전 누적, >0이면 지수 가중)
    ):
        self.lambda_max_estimator = lambda_max_estimator
        self.Φ = potential_function

        self.theta_enter = theta_enter
        self.theta_exit = theta_exit
        self.alpha = alpha
        self.max_tightening = max_tightening
        self.recovery_steps = recovery_steps

        self.proj_lr = proj_lr
        self.proj_steps_max = proj_steps_max
        self.proj_steps_min = proj_steps_min
        self.grad_clip_max_norm = grad_clip_max_norm
        self.cooldown_k = cooldown_k
        self.expl_decay_k = expl_decay_k
        self.phi_improve_tol = phi_improve_tol
        self.lambda_clip_margin = lambda_clip_margin

        self.resonance_decay = resonance_decay
        self.t_lambda = t_lambda

        # Safe Mode 상태 (hard reset 대상)
        self.safe_mode: bool = False
        self.safe_mode_counter: int = 0
        self.current_constraint_weight: float = 1.0
        self._peak_phi: float = 0.0
        self._m: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

        # FWR 상태 (soft reset: R만 초기화, T 보존)
        self.R: float = 1.0
        self.T: float = 0.0

    def reset(self, hard: bool = False) -> None:
        """
        hard=False (기본): Safe Mode 관련 상태만 초기화. T는 보존.
        hard=True: R, T 포함 전체 초기화.
        """
        self.safe_mode = False
        self.safe_mode_counter = 0
        self.current_constraint_weight = 1.0
        self._peak_phi = 0.0
        self._m = self._v = None

        if hard:
            self.R = 1.0
            self.T = 0.0

    def _ensure_adam_buffers(self, ref: torch.Tensor) -> None:
        if (
            self._m is None
            or self._m.shape != ref.shape
            or self._m.device != ref.device
        ):
            self._m = torch.zeros_like(ref)
            self._v = torch.zeros_like(ref)

    def compute_stability_score(
        self,
        system_jacobian: Optional[torch.Tensor],
        state: torch.Tensor,
    ) -> float:
        if system_jacobian is None:
            return 0.70
        try:
            λ = float(self.lambda_max_estimator(system_jacobian, state))
            if not math.isfinite(λ):
                logger.warning(f"Non-finite λ_max: {λ}")
                return 0.04
            λ = max(0.0, min(λ, self.lambda_clip_margin))
            return math.exp(-self.alpha * λ)
        except Exception as e:
            logger.error(f"Stability computation failed: {e}")
            return 0.04

    def project_with_adaptive_step(
        self,
        state: torch.Tensor,
        S_stab: float,
    ) -> torch.Tensor:
        """
        Φ(x) 최소화 방향으로 state를 투영.
        no_grad 데코레이터 제거 — 내부에서 grad context를 명시적으로 분리.
        """
        criticality = max(0.0, (self.theta_enter - S_stab) / self.theta_enter)
        n_steps = int(
            self.proj_steps_min
            + (self.proj_steps_max - self.proj_steps_min) * (criticality ** 1.5)
        )
        effective_lr = self.proj_lr * (1.0 + 0.6 * criticality)

        # 외부 grad 흐름과 완전히 분리된 새 텐서
        x = state.detach().clone().requires_grad_(True)
        self._ensure_adam_buffers(x)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_phi = float("inf")
        best_x: Optional[torch.Tensor] = None
        prev_phi = float("inf")

        for step in range(n_steps):
            if x.grad is not None:
                x.grad.zero_()

            phi = self.Φ(x) * self.current_constraint_weight
            curr_phi = phi.item()

            if not math.isfinite(curr_phi):
                logger.warning("Non-finite Φ → early stop")
                break

            if curr_phi < best_phi - 1e-6:
                best_phi = curr_phi
                best_x = x.detach().clone()

            if step >= 3 and (prev_phi - curr_phi) < self.phi_improve_tol:
                break
            prev_phi = curr_phi

            phi.backward()

            with torch.no_grad():
                if x.grad is None:
                    break

                torch.nn.utils.clip_grad_norm_([x], self.grad_clip_max_norm)

                self._m.mul_(beta1).add_(x.grad, alpha=1 - beta1)
                self._v.mul_(beta2).addcmul_(x.grad, x.grad, value=1 - beta2)

                m_hat = self._m / (1 - beta1 ** (step + 1))
                v_hat = self._v / (1 - beta2 ** (step + 1))

                x.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-effective_lr)

        self._peak_phi = max(self._peak_phi, best_phi)
        return best_x if best_x is not None else state.detach().clone()

    def _update_fwr_state(self, S_stab: float) -> None:
        """
        R(t), T(t) 업데이트.
        - Crisis: R 붕괴 (T가 높을수록 완충)
        - Normal/Recovery: R 회복
        - T: 단조증가 보장 (t_lambda=0이면 완전 누적, >0이면 지수 가중)
        """
        if S_stab < self.theta_enter:
            # T_influence: 로그 스케일로 포화 방지
            T_influence = 1.0 + math.log1p(self.T)
            exponent = 1.0 / T_influence
            decay = 0.78 ** exponent
            self.R *= decay
        else:
            self.R = self.resonance_decay * self.R + (1.0 - self.resonance_decay)

        self.R = max(0.05, min(1.0, self.R))

        # T 누적 — 단조증가 보장
        if self.t_lambda > 0:
            # 지수 가중: 오래된 기억 할인
            self.T = math.exp(-self.t_lambda) * self.T + self.R
        else:
            # 완전 누적
            self.T += self.R

    def _manage_safe_mode(self, S_stab: float) -> None:
        self._update_fwr_state(S_stab)

        if S_stab < self.theta_enter:
            if not self.safe_mode:
                logger.warning(
                    f"[FWR] SAFE MODE ACTIVATED | S={S_stab:.3f} | "
                    f"R={self.R:.3f} | T={self.T:.2f}"
                )
                self.safe_mode = True
                self.safe_mode_counter = self.recovery_steps

            ramp = min(
                1.0,
                (self.recovery_steps - self.safe_mode_counter + 1) / 15.0,
            )
            self.current_constraint_weight = (
                1.0 + (self.max_tightening - 1.0) * ramp
            )

        elif self.safe_mode:
            if S_stab > self.theta_exit:
                self.safe_mode_counter -= 1

            progress = max(
                0.0, 1.0 - self.safe_mode_counter / self.recovery_steps
            )
            decay = math.exp(-self.cooldown_k * progress)
            self.current_constraint_weight = (
                1.0 + (self.max_tightening - 1.0) * (1.0 - decay)
            )

            if self.safe_mode_counter <= 0:
                logger.info(
                    f"[FWR] SAFE MODE RELEASED | peak Φ={self._peak_phi:.3e} | "
                    f"R={self.R:.3f} | T={self.T:.2f}"
                )
                # soft reset: T 보존
                self.reset(hard=False)

    def __call__(
        self,
        current_state: torch.Tensor,
        exploration_rate: float,
        system_jacobian: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float, dict]:
        """
        Returns:
            new_state          : projected state (safe mode) or current_state
            new_exploration    : reduced exploration rate in safe mode
            constraint_weight  : current Φ constraint multiplier
            status             : {"S_stab", "R", "T", "constraint_weight", "safe_mode"}
        """
        S = self.compute_stability_score(system_jacobian, current_state)
        self._manage_safe_mode(S)

        status = {
            "S_stab": S,
            "R": self.R,
            "T": self.T,
            "constraint_weight": self.current_constraint_weight,
            "safe_mode": self.safe_mode,
        }

        if not self.safe_mode:
            return current_state, exploration_rate, 1.0, status

        projected = self.project_with_adaptive_step(current_state, S)
        crit = max(0.0, (self.theta_enter - S) / self.theta_enter)
        reduced_expl = exploration_rate * math.exp(-self.expl_decay_k * crit)

        return projected, reduced_expl, self.current_constraint_weight, status
