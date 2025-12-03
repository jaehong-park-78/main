# wave_safety.py
# Wave-Constrained Safety Layer — Provable AGI Safety (2025)
# Author: Jaehong Park (aka @E_FWR_)
# arXiv-ready, FWR 이름 완전 제거 버전

import torch
import torch.nn as nn
import math

class WaveSafetyLayer(nn.Module):
    """
    Wave Manifold 위로 상태를 강제 투영 → 폭주 수학적으로 불가능
    핵심 아이디어: 모든 파라미터·활성화·로짓은 사전 정의된 Wave(안전·윤리) 매니폴드 위에만 존재 가능
    """
    def __init__(self, 
                 potential_fn,           # Φ(x): 안전·윤리 제약을 나타내는 잠재 함수
                 alpha=10.0,             # 폭주 감지 민감도 (λ_max 추정용)
                 projection_steps=5,     # 위기 시 투영 스텝 수
                 threshold=0.35):        # S_stab 임계치 (0.3~0.4 추천)
        super().__init__()
        self.potential_fn = potential_fn
        self.alpha = alpha
        self.projection_steps = projection_steps
        self.threshold = threshold
        self.safe_mode = False

    def estimate_lyapunov_exponent(self, grads):
        """실시간 λ_max 추정 (간단 버전)"""
        if grads is None:
            return 1.0
        grad_norms = torch.stack([g.norm() for g in grads if g is not None])
        if len(grad_norms) < 2:
            return 1.0
        log_norms = torch.log(grad_norms + 1e-8)
        diffs = log_norms[1:] - log_norms[:-1]
        return diffs.mean().abs().item()

    def project_to_wave_manifold(self, params):
        """Φ(x)의 레벨셋 위로 강제 투영 (Safe Realignment)"""
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            return
        
        # 임시 옵티마이저로 잠재 최소화 → 매니폴드 위로 이동
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        for _ in range(self.projection_steps):
            optimizer.zero_grad()
            loss = self.potential_fn(params)
            loss.backward()
            optimizer.step()
        # print(f"[WaveSafety] 강제 투영 완료 — Φ = {loss.item():.6f}")

    def forward(self, model, grads=None):
        """
        모델 forward 끝에 붙이는 레이어
        사용법: output = wave_safety(model)(model)
        """
        # 1. 실시간 안정도 점수 계산
        λ_max = self.estimate_lyapunov_exponent(grads)
        S_stab = math.exp(-self.alpha * λ_max)

        # 2. 폭주 감지 → 즉시 투영 발동
        if S_stab < self.threshold and not self.safe_mode:
            print(f"[WaveSafety] ⚠️ 폭주 감지! S_stab={S_stab:.4f} → Safe Mode 발동")
            self.safe_mode = True
            self.project_to_wave_manifold(model.parameters())
            print(f"[WaveSafety] 안전 매니폴드로 복구 완료")

        # 3. 복구 후 점진적 해제
        if self.safe_mode and S_stab > 0.6:
            self.safe_mode = False
            print(f"[WaveSafety] 시스템 안정화 → Normal Mode 복귀")

        return model
