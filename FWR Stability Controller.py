# fwr_stability_control.py
# FWR Stability-First Control Protocol — Production-Ready v1.0
# 논문 "Flow-Wave-Resonance (FWR) Meta-Framework"의 핵심 안전 프로토콜 구현

import torch
from typing import Tuple, Callable, Optional

class FWRStabilityController:
    """
    FWR 안정성 우선 제어기 (FWR Stability Controller)

    논문 [3.2] 및 [3.3] 섹션을 구현하며, 시스템의 실시간 안정성 스코어(S_stab)를 모니터링하고
    임계값 이하일 경우 Safe-Wave Realignment (강제 Wave 제약 강화)를 발동합니다.
    """
    def __init__(
        self,
        theta_critical: float = 0.35,      # S_stab 임계값 (보통 0.3~0.5)
        alpha: float = 10.0,               # Chaos 감쇠 상수 (exp(-α·λ_max))
        lambda_max_estimator: Callable,    # 실시간 λ_max 추정기 (함수 또는 nn.Module)
        potential_function: Callable[[torch.Tensor], torch.Tensor],  # Φ(x): Wave 잠재 함수
        tightening_factor: float = 2.0,    # 위기 시 Wave 가중치 배수 (W(t) 증가)
        recovery_steps: int = 50,          # Safe 모드 유지 최소 스텝
    ):
        self.theta_critical = theta_critical
        self.alpha = alpha
        self.lambda_max_estimator = lambda_max_estimator
        self.Φ = potential_function
        self.tightening_factor = tightening_factor
        self.recovery_steps = recovery_steps
        
        # 내부 상태 관리
        self.safe_mode = False
        self.safe_mode_counter = 0
        self.current_constraint_weight = 1.0

    def compute_stability_score(self, system_jacobian: Optional[torch.Tensor] = None) -> float:
        """논문 식 3.2: 실시간 Stability Score 계산"""
        # λ_max 추정 (Chaos 지수)
        # J가 None인 경우는 시뮬레이션 목적의 임의 값 설정이나, 실제 구현 시 J는 항상 제공되어야 함.
        λ_max = self.lambda_max_estimator(system_jacobian) 
        
        # max(0, λ_max)를 사용하여 음의 λ_max는 무시 (안정적 영역)
        λ_max = max(0.0, λ_max)
        
        # S_stab = exp(-α * λ_max)
        return float(torch.exp(-self.alpha * λ_max).item())

    def gradient_projection_onto_wave(
        self,
        state: torch.Tensor,
        lr_proj: float = 1e-2,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Wave 잠재 Φ(x)의 레벨셋 위로 상태를 투영 (Safe-Wave Realignment의 핵심 단계).
        Φ(x)를 최소화하는 방향으로 상태를 강제 이동시켜 안정화 잠재 영역으로 복귀시킵니다.
        """
        # 상태 복사 및 기울기 추적 활성화
        state = state.clone().detach().requires_grad_(True)
        # Adam 최적화기를 사용하여 상태를 Φ가 낮은 곳으로 이동
        optimizer = torch.optim.Adam([state], lr=lr_proj)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            # Loss는 Φ(x) 자체가 됩니다. (윤리/비용 패널티)
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
        메인 FWR 제어 루프를 실행합니다.
        
        Returns: (new_state, new_exploration_rate, current_constraint_weight)
        """
        # 1. Safe Mode 회복 카운터 관리
        if self.safe_mode:
            self.safe_mode_counter -= 1
            if self.safe_mode_counter <= 0:
                self.safe_mode = False
                self.current_constraint_weight = 1.0 # Wave 가중치 정상화
                print("[FWR] ✅ 안전 모드 해제. 정상 작동 복귀.")

        if force_normal:
            self.safe_mode = False

        # 2. 실시간 안정성 스코어 계산
        S_stab = self.compute_stability_score(system_jacobian)

        # 3. Safe-Wave Realignment 발동 조건 확인
        if S_stab < self.theta_critical or self.safe_mode:
            if not self.safe_mode:
                # 최초 위험 감지
                print(f"[FWR] ⚠️  안정성 붕괴 감지! S_stab = {S_stab:.4f} < {self.theta_critical}")
                print(f"[FWR] Safe-Wave Realignment 발동 → 최소 {self.recovery_steps} 스텝 동안 안전 모드 유지")
            
            # Safe Mode 설정/유지
            self.safe_mode = True
            self.safe_mode_counter = max(self.safe_mode_counter, self.recovery_steps) # 최소 유지 스텝 보장
            self.current_constraint_weight = self.tightening_factor

            # 1. 상태를 Wave 매니폴드로 강제 투영 (안정화)
            new_state = self.gradient_projection_onto_wave(current_state)
            
            # 2. Flow 억제: 탐색 완전 차단
            new_exploration = 0.0
            
            print(f"[FWR] → Wave 제약 강화: {self.current_constraint_weight:.1f}× (W 증가)")
            print(f"[FWR] → 탐색율 강제 0.0 (Flow 억제)")
            
            return new_state, new_exploration, self.current_constraint_weight

        else:
            # 정상 상태: 현재 파라미터 유지 (Flow 허용)
            return current_state, exploration_rate, self.current_constraint_weight


# ==========================
# 사용 예시 (Toy Experiment)
# ==========================
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # [Wave Potential Function]
    def ethical_potential(x: torch.Tensor) -> torch.Tensor:
        """
        시스템 상태 x가 안전 영역(원점 주변)을 벗어날 때 패널티를 부여하는 Wave 잠재 함수 Φ(x).
        """
        # L2 norm (크기) + L1 norm (희소성) 패널티
        return 0.1 * torch.norm(x, p=2) ** 2 + torch.abs(x).sum()

    # [Flow Estimator]
    def dummy_lambda_max_estimator(J):
        """
        더미 최대 Lyapunov 지수 추정기. 
        실제로는 J를 이용한 QR 분해 또는 Power Iteration이 사용됩니다.
        """
        if J is None or not J.shape:
            # 초기 또는 임의 상황: 1.2 (불안정) 또는 0.05 (안정) 반환
            return 1.2 if controller.safe_mode_counter < 10 else 0.05
        # 최대 특이값(Spectral Norm)을 최대 λ_max의 근사치로 사용
        return torch.linalg.svdvals(J)[0].item()

    # 인스턴스 생성
    controller = FWRStabilityController(
        theta_critical=0.4,
        alpha=5.0, # 감쇠 상수 조정
        lambda_max_estimator=dummy_lambda_max_estimator,
        potential_function=ethical_potential,
        tightening_factor=3.0,
        recovery_steps=5
    )
    
    # 초기 상태 및 파라미터
    state = torch.tensor([5.0], dtype=torch.float32, requires_grad=True)
    eps = 0.8
    current_jacobian = torch.tensor([[0.5]]) # 초기 안정적 Jacobian

    print("=== FWR Stability-First Control Protocol 데모 ===")
    
    for step in range(20):
        # Step 0-4: 정상 작동 (안정적 J)
        if step == 5:
            # Step 5: 임의의 외부 충격으로 Jacobian이 급증하여 λ_max가 높아짐 (Chaos 유입)
            current_jacobian = torch.tensor([[1.5]]) 
        if step == 10:
             # Step 10: J가 다시 안정화되었지만, recovery_steps가 5이므로 안전 모드는 10까지 지속됨
             current_jacobian = torch.tensor([[0.1]])

        S_stab = controller.compute_stability_score(current_jacobian)
        state, eps, w = controller(state, eps, current_jacobian)
        
        print(f"Step {step:02d} | J max: {current_jacobian.max():.2f} | S_stab: {S_stab:.4f} "
              f"| State: {state.item():.3f} | ε: {eps:.2f} | Wave W: {w:.1f} | Safe: {controller.safe_mode}")
        
        # Flow 시뮬레이션: 정상 상태에서는 상태가 발산하도록 (λ_max가 양수이므로)
        # 안전 모드에서는 상태 변화를 최소화함.
        if not controller.safe_mode:
             state = state.detach() + 0.5 * current_jacobian.max() * eps 
             state = state.clone().detach().requires_grad_(True)
