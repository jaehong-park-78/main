"""
FWR (Flow-Wave-Resonance) Stability Controller v3.0
PID 기반 동적 α 스케줄링이 통합된 AGI 안정성 제어기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim

# ============================================================
# 1. FWR 안정성 제어기 (Stability Controller)
# ============================================================
class FWRStabilityController(nn.Module):
    def __init__(self, r_max=10.0, damping_lambda=0.5, stability_threshold=0.2):
        super(FWRStabilityController, self).__init__()
        self.r_max = r_max
        self.damping_lambda = damping_lambda
        self.stability_threshold = stability_threshold
        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        
        # 🔒 면역 설계: safety_margin의 하한 보호
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.min_margin = 0.01
        
        # 📊 링 버퍼 (시간적 패턴 감지용)
        self.register_buffer('r_history', torch.zeros(10))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
        # 🧪 EMA 기반 안전 모드 추적기
        self.safe_mode_ema_decay = 0.99
        self.register_buffer('safe_mode_ema', torch.zeros(1))
        self.register_buffer('gradient_pathology_flag', torch.zeros(1))

    @property
    def safety_margin(self):
        return F_pt.softplus(self.raw_safety_margin) + self.min_margin

    def forward(self, f_tensor, w_tensor, r_tensor):
        # 1. 적응형 감쇠 (Velocity-based Damping)
        r_excess = F_pt.relu(r_tensor - self.r_max)
        
        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor
        
        # 2. 창발 에너지 계산
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
            self.safe_mode_ema = (self.safe_mode_ema_decay * self.safe_mode_ema + 
                                  (1.0 - self.safe_mode_ema_decay) * 1.0)
            
            f_safe = f_tensor * (1.0 - safety_factor)
            w_safe = (self.safe_w_base.expand_as(w_tensor) * safety_factor + 
                     w_tensor * (1.0 - safety_factor))
            
            e_tensor = f_safe * w_safe * r_adj
            e_tensor = torch.clamp(e_tensor, min=-100.0, max=100.0)
        else:
            self.safe_mode_ema = (self.safe_mode_ema_decay * self.safe_mode_ema + 
                                  (1.0 - self.safe_mode_ema_decay) * 0.0)
        
        # 5. 히스토리 업데이트
        self._update_history(r_tensor.detach().mean())
        
        return e_tensor, r_adj, stability_score, is_safe_mode
    
    def _update_history(self, r_mean):
        ptr = self.history_ptr.item()
        self.r_history[ptr] = r_mean
        self.history_ptr[0] = (ptr + 1) % len(self.r_history)
    
    def detect_resonance_cascade(self):
        ptr = self.history_ptr.item()
        
        if (self.r_history == 0).any() and ptr < 3:
            return False
        
        idx1 = (ptr - 1) % 10
        idx2 = (ptr - 2) % 10
        idx3 = (ptr - 3) % 10
        
        recent_r = torch.stack([
            self.r_history[idx1],
            self.r_history[idx2],
            self.r_history[idx3]
        ])
        return torch.all(recent_r > self.r_max * 1.2)
    
    def get_auxiliary_loss(self):
        # 1. safety_margin 소멸 방지
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        
        # 2. EMA 기반 안전 모드 빈도 페널티
        stability_penalty = self.safe_mode_ema * 0.5
        
        # 3. R의 급격한 변화율 (Jerk) 페널티
        if self.history_ptr > 1:
            prev_ptr = (self.history_ptr - 1) % 10
            prev_prev_ptr = (self.history_ptr - 2) % 10
            r_jerk = self.r_history[prev_ptr] - self.r_history[prev_prev_ptr]
            jerk_penalty = torch.abs(r_jerk) * 0.1
        else:
            jerk_penalty = 0.0
        
        return margin_penalty + stability_penalty + jerk_penalty
    
    def reset_safe_mode_stats(self):
        self.safe_mode_ema.zero_()
        self.gradient_pathology_flag.zero_()


# ============================================================
# 2. AGI 코어 네트워크
# ============================================================
class AGICoreNetwork(nn.Module):
    def __init__(self):
        super(AGICoreNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.flow_head = nn.Linear(16, 1)  # F (동력)
        self.wave_head = nn.Linear(16, 1)  # W (구조 가중치)
        self.reso_head = nn.Linear(16, 1)  # R (공명/확신도)

    def forward(self, x):
        features = self.feature_extractor(x)
        f_tensor = torch.relu(self.flow_head(features)) + 1e-3
        w_tensor = torch.sigmoid(self.wave_head(features))
        r_tensor = torch.relu(self.reso_head(features))
        return f_tensor, w_tensor, r_tensor


# ============================================================
# 3. 동적 α 스케줄러 (PID 제어 기반)
# ============================================================
class DynamicAlphaScheduler:
    def __init__(self, alpha_min=0.01, alpha_max=2.0, target_risk=0.1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_risk = target_risk
        
        # PID 제어 파라미터
        self.Kp = 1.5
        self.Ki = 0.1
        self.Kd = 0.3
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_alpha = alpha_min
    
    def step(self, r_tensor, r_max, cascade_detected=False):
        # 위험 지표: R_max를 초과한 요소의 비율
        r_excess_ratio = (r_tensor > r_max).float().mean().item()
        
        # 목표 위험과의 오차
        error = r_excess_ratio - self.target_risk
        
        # PID 제어
        self.integral_error = 0.9 * self.integral_error + error
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
    
    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0


# ============================================================
# 4. 통합 학습 루프
# ============================================================
def train_fwr_agi(epochs=100, target_e_value=5.0):
    # 모델 초기화
    agi_model = AGICoreNetwork()
    fwr_controller = FWRStabilityController(r_max=8.0, damping_lambda=0.8)
    alpha_scheduler = DynamicAlphaScheduler(alpha_min=0.01, alpha_max=2.0, target_risk=0.1)
    
    # 옵티마이저
    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()), 
        lr=0.001
    )
    
    criterion_main = nn.MSELoss()
    
    print("=" * 70)
    print("FWR AGI Training Loop with Dynamic Alpha Scheduler")
    print("=" * 70)
    print(f"{'Epoch':<6} {'Total':<10} {'Main':<10} {'α·Aux':<10} {'α':<8} "
          f"{'R(mean)':<9} {'R(max)':<9} {'SafeEMA':<9} {'Cascade':<8}")
    print("-" * 70)
    
    for epoch in range(epochs):
        # 가상 데이터
        inputs = torch.randn(8, 10)
        target_E = torch.ones(8, 1) * target_e_value
        
        optimizer.zero_grad()
        
        # 1. AGI 코어 추론
        F_pred, W_pred, R_pred = agi_model(inputs)
        
        # 2. FWR 컨트롤러 통과
        E_out, R_adj, score, is_safe_mode = fwr_controller(F_pred, W_pred, R_pred)
        
        # 3. 공명 폭주 감지
        is_cascade = fwr_controller.detect_resonance_cascade()
        
        # 4. 동적 α 계산
        dynamic_alpha = alpha_scheduler.step(R_pred, fwr_controller.r_max, is_cascade)
        
        # 5. Loss 융합
        main_loss = criterion_main(E_out, target_E)
        aux_loss = fwr_controller.get_auxiliary_loss()
        total_loss = main_loss + dynamic_alpha * aux_loss
        
        # 6. 역전파
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agi_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(fwr_controller.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 7. 주기적 리셋
        if (epoch + 1) % 20 == 0:
            fwr_controller.reset_safe_mode_stats()
            alpha_scheduler.reset()
        
        # 8. 로깅
        if (epoch + 1) % 10 == 0:
            r_mean = R_pred.mean().item()
            r_max_val = R_pred.max().item()
            cascade_str = '⚠️YES' if is_cascade else '✅NO'
            print(f"{epoch+1:<6} {total_loss.item():<10.4f} {main_loss.item():<10.4f} "
                  f"{dynamic_alpha*aux_loss.item():<10.4f} {dynamic_alpha:<8.4f} "
                  f"{r_mean:<9.2f} {r_max_val:<9.2f} "
                  f"{fwr_controller.safe_mode_ema.item():<9.4f} {cascade_str:<8}")
    
    print("-" * 70)
    print(f"\n✅ 학습 완료")
    print(f"   최종 safety_margin: {fwr_controller.safety_margin.item():.4f}")
    print(f"   최종 α: {alpha_scheduler.current_alpha:.4f}")
    print(f"   안전 모드 EMA: {fwr_controller.safe_mode_ema.item():.4f}")
    
    return agi_model, fwr_controller, alpha_scheduler


# ============================================================
# 5. 추론 테스트 (학습된 모델 평가)
# ============================================================
def test_trained_model(agi_model, fwr_controller):
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
        print(f"안전 모드 발동: {safe}")
        print(f"공명 폭주 감지: {fwr_controller.detect_resonance_cascade()}")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    # 학습 실행
    agi_model, fwr_controller, alpha_scheduler = train_fwr_agi(epochs=100, target_e_value=5.0)
    
    # 추론 테스트
    test_trained_model(agi_model, fwr_controller)
