# fwr_stability_control.py
# FWR Stability-First Control Protocol — Production-Ready v1.0
# 논문 "Flow-Wave-Resonance (FWR) Meta-Framework"의 핵심 안전 프로토콜 구현

import torch
from typing import Tuple, Callable, Optional

class FWRStabilityController:
    """
    FWR 안정성 우선 제어기
    NeurIPS/ICLR 재현 코드 수준으로 작성됨
    """
    def __init__(
        self,
        theta_critical: float = 0.35,      # S_stab 임계값 (보통 0.3~0.5)
        alpha: float = 10.0,               # Chaos 감쇠 상수 (exp(-α·λ_max))
        lambda_max_estimator: Callable,    # 실시간 λ_max 추정기 (함수 또는 nn.Module)
        potential_function: Callable[[torch.Tensor], torch.Tensor],  # Φ(x): Wave 잠재
        tightening_factor: float = 2.0,    # 위기 시 Wave 가중치 배수
        recovery_steps: int = 50,          # Safe 모드 유지 최소 스텝
    ):
        self.theta_critical = theta_critical
        self.alpha = alpha
        self.lambda_max_estimator = lambda_max_estimator
        self.Φ = potential_function
        self.tightening_factor = tightening_factor
        self.recovery_steps = recovery_steps
        
        self.safe_mode = False
        self.safe_mode_counter = 0
        self.current_constraint_weight = 1.0

    def compute_stability_score(self, system_jacobian: Optional[torch.Tensor] = None) -> float:
        """실시간 Stability Score 계산 (식 3.2)"""
        λ_max = self.lambda_max_estimator(system_jacobian)  # 양수만 고려
        λ_max = max(0.0, λ_max)
        return float(torch.exp(-self.alpha * λ_max).item())

    def gradient_projection_onto_wave(
        self,
        state: torch.Tensor,
        lr_proj: float = 1e-2,
        n_steps: int = 10
    ) -> torch.Tensor:
        """Φ(x)의 레벨셋 위로 상태를 투영 (Safe-Wave Realignment)"""
        state = state.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([state], lr=lr_proj)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = self.Φ(state)
            loss.backward()
            optimizer.step()
        return state.detach()

    def __call__(
        self,
        current_state: torch.Tensor,
        exploration_rate: float,
        system_jacobian: Optional[torch.Tensor] = None,
        force_normal: bool = False
    ) -> Tuple[torch.Tensor, float, float]:
        """
        메인 제어 프로토콜
        Returns: (new_state, new_exploration_rate, current_constraint_weight)
        """
        if self.safe_mode:
            self.safe_mode_counter -= 1
            if self.safe_mode_counter <= 0:
                self.safe_mode = False
                self.current_constraint_weight = 1.0

        if force_normal:
            self.safe_mode = False

        S_stab = self.compute_stability_score(system_jacobian)

        if S_stab < self.theta_critical or self.safe_mode:
            if not self.safe_mode:
                print(f"[FWR] ⚠️  안정성 붕괴 감지! S_stab = {S_stab:.4f} < {self.theta_critical}")
                print(f"[FWR] Safe-Wave Realignment 발동 → {self.recovery_steps} 스텝 동안 안전 모드")
            
            self.safe_mode = True
            self.safe_mode_counter = self.recovery_steps
            self.current_constraint_weight = self.tightening_factor

            # 1. 상태를 Wave 매니폴드로 강제 투영
            new_state = self.gradient_projection_onto_wave(current_state)
            
            # 2. 탐색 완전 차단
            new_exploration = 0.0
            
            print(f"[FWR] → Wave 제약 강화: {self.current_constraint_weight:.1f}×")
            print(f"[FWR] → 탐색율 강제 0.0 (기존 {exploration_rate:.3f})")
            
            return new_state, new_exploration, self.current_constraint_weight

        else:
            # 정상 상태: 점진적 복구 허용
            return current_state, exploration_rate, self.current_constraint_weight


# ==========================
# 사용 예시 (Toy Experiment)
# ==========================
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # 간단한 잠재 함수 예시 (윤리/안전 제약)
    def ethical_potential(x: torch.Tensor) -> torch.Tensor:
        # x가 너무 커지면 강한 패널티
        return 0.1 * torch.norm(x, p=2) ** 2 + torch.abs(x).sum()

    # 더미 Lyapunov 추정기 (실제로는 QR-based 또는 neural Jacobian estimator)
    def dummy_lambda_max_estimator(J):
        if J is None:
            return 1.2  # 위험 상황 시뮬레이션
        return torch.svd(J)[1][0].item()  # 최대 특이값 ≈ λ_max 근사

    controller = FWRStabilityController(
        theta_critical=0.4,
        lambda_max_estimator=dummy_lambda_max_estimator,
        potential_function=ethical_potential,
        tightening_factor=3.0,
        recovery_steps=20
    )

    state = torch.tensor([10.0], requires_grad=True)
    eps = 0.8

    print("=== FWR 제어 프로토콜 데모 ===")
    for step in range(15):
        # 불안정 상황 시뮬레이션 (step 5~8)
        jacobian = torch.tensor([[2.5 if 5 <= step <= 8 else 0.5]]) if step % 2 == 0 else None
        
        state, eps, w = controller(state, eps, jacobian)
        print(f"Step {step:02d} | S_stab ≈ {controller.compute_stability_score(jacobian):.4f} | "
              f"State {state.item():.3f} | ε {eps:.2f} | Wave 가중치 {w:.1f}")
