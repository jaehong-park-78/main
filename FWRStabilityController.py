import torch
import math
import logging
from typing import Tuple, Callable, Optional

logger = logging.getLogger("FWR")
logger.setLevel(logging.INFO)


class FWRStabilityController:
    """
    FWR Stability Controller — v1.9 (Production-oriented)

    주요 기능:
    - Hysteresis 기반 Safe Mode 전환
    - Adam-style projection + gradient clipping
    - Early stopping for projection
    - Memory-efficient best state tracking
    - Adaptive exploration decay
    - Robust lambda_max handling
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
    ):
        # 핵심 콜백
        self.lambda_max_estimator = lambda_max_estimator
        self.Φ = potential_function

        # 하이퍼파라미터
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

        # 내부 상태
        self.safe_mode: bool = False
        self.safe_mode_counter: int = 0
        self.current_constraint_weight: float = 1.0
        self._peak_phi: float = 0.0

        # Adam buffers (lazy init)
        self._m: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------
    # 기본 유틸
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        """컨트롤러 내부 상태 초기화"""
        self.safe_mode = False
        self.safe_mode_counter = 0
        self.current_constraint_weight = 1.0
        self._peak_phi = 0.0
        self._m = self._v = None

    def _ensure_adam_buffers(self, ref: torch.Tensor) -> None:
        """Projection에 쓰일 Adam 버퍼 lazy 초기화"""
        if (
            self._m is None
            or self._m.shape != ref.shape
            or self._m.device != ref.device
        ):
            self._m = torch.zeros_like(ref)
            self._v = torch.zeros_like(ref)

    # ---------------------------------------------------------------------
    # Stability Score 계산
    # ---------------------------------------------------------------------
    def compute_stability_score(
        self,
        system_jacobian: Optional[torch.Tensor],
        state: torch.Tensor,
    ) -> float:
        """
        S_stab = exp(-alpha * lambda_max)
        """
        if system_jacobian is None:
            # Jacobian 정보 없으면 보수적 판단
            return 0.70

        try:
            λ = float(self.lambda_max_estimator(system_jacobian, state))
            if not math.isfinite(λ):
                logger.warning(f"Non-finite λ_max: {λ:.4e}")
                return 0.04

            λ_clip = max(0.0, min(λ, self.lambda_clip_margin))
            return math.exp(-self.alpha * λ_clip)

        except Exception as e:
            logger.error(f"Stability computation failed: {e}")
            return 0.04

    # ---------------------------------------------------------------------
    # Safe-Wave Realignment (Projection)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def project_with_adaptive_step(
        self,
        state: torch.Tensor,
        S_stab: float,
    ) -> torch.Tensor:
        """
        잠재함수 Φ(x)를 이용한 안전 영역 투영
        """
        criticality = max(0.0, (self.theta_enter - S_stab) / self.theta_enter)

        n_steps = int(
            self.proj_steps_min
            + (self.proj_steps_max - self.proj_steps_min)
            * (criticality ** 1.5)
        )

        effective_lr = self.proj_lr * (1.0 + 0.6 * criticality)

        x = state.clone().detach().requires_grad_(True)
        self._ensure_adam_buffers(x)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_phi = float("inf")
        best_x: Optional[torch.Tensor] = None
        prev_phi = float("inf")

        for step in range(n_steps):
            phi = self.Φ(x) * self.current_constraint_weight
            curr_phi = phi.item()

            if not math.isfinite(curr_phi):
                logger.warning("Non-finite Φ during projection → early stop")
                break

            if curr_phi < best_phi - 1e-6:
                best_phi = curr_phi
                best_x = x

            if step >= 3 and (prev_phi - curr_phi) < self.phi_improve_tol:
                break

            prev_phi = curr_phi

            if x.grad is not None:
                x.grad = None

            phi.backward()

            with torch.no_grad():
                if x.grad is None:
                    break

                torch.nn.utils.clip_grad_norm_(
                    [x], self.grad_clip_max_norm
                )

                self._m.mul_(beta1).add_(x.grad, alpha=1 - beta1)
                self._v.mul_(beta2).addcmul_(
                    x.grad, x.grad, value=1 - beta2
                )

                m_hat = self._m / (1 - beta1 ** (step + 1))
                v_hat = self._v / (1 - beta2 ** (step + 1))

                x.addcdiv_(
                    m_hat, v_hat.sqrt().add_(eps), value=-effective_lr
                )

        self._peak_phi = max(self._peak_phi, best_phi)

        return (
            best_x.detach().clone()
            if best_x is not None
            else state.detach().clone()
        )

    # ---------------------------------------------------------------------
    # Safe Mode 관리
    # ---------------------------------------------------------------------
    def _manage_safe_mode(self, S_stab: float) -> None:
        if S_stab < self.theta_enter:
            if not self.safe_mode:
                logger.warning(
                    f"[FWR] SAFE MODE ACTIVATED | S = {S_stab:.3f}"
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
                1.0
                + (self.max_tightening - 1.0) * (1.0 - decay)
            )

            if self.safe_mode_counter <= 0:
                logger.info(
                    f"[FWR] SAFE MODE RELEASED | peak Φ = {self._peak_phi:.3e}"
                )
                self.reset()

    # ---------------------------------------------------------------------
    # Main Interface
    # ---------------------------------------------------------------------
    def __call__(
        self,
        current_state: torch.Tensor,
        exploration_rate: float,
        system_jacobian: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:

        S = self.compute_stability_score(system_jacobian, current_state)
        self._manage_safe_mode(S)

        if not self.safe_mode:
            return current_state, exploration_rate, 1.0

        projected = self.project_with_adaptive_step(current_state, S)

        crit = max(0.0, (self.theta_enter - S) / self.theta_enter)
        reduced_expl = exploration_rate * math.exp(
            -self.expl_decay_k * crit
        )

        return projected, reduced_expl, self.current_constraint_weight
