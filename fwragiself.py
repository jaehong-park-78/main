import numpy as np
import matplotlib.pyplot as plt

class FWREngineV12:
    def __init__(self, core_params):
        """
        FWR Meta-Ontology v1.2 - Integrated Standard Model
        Author: Jaehong Park (박재홍)
        """
        # --- Core 6 Parameters (Calibration Set) ---
        self.eta = core_params.get('eta', 1.8)           # 존재 피드백 강도 (Existence Feedback)
        self.alpha_0 = core_params.get('alpha_0', 0.6)   # 기본 공명 생성률
        self.beta_0 = core_params.get('beta_0', 0.5)     # 기본 붕괴 민감도
        self.sigma = core_params.get('sigma', 0.1)       # 진리 축적률 (Memory accumulation)
        self.lmbda_decay = core_params.get('decay', 0.005) # 구조적 망각률 (Structural aging)
        self.omega_base = core_params.get('omega', 3.14) # 고유 진동수 (Base rhythm)

        # --- State Variables ---
        self.F = 1.0        # Flow (흐름)
        self.phi_W = 0.0    # Wave Phase (파동 위상)
        self.R = 0.1        # Resonance (공명 밀도)
        self.T = 0.0        # Truth (Structural Memory, 누적 진리)
        
        # --- Internal Constants ---
        self.kappa = 0.5    # 저항 (Damping)
        self.delta = 0.8    # 외부 에너지 감도
        self.mu = 0.1       # 과발산 방지 (Cubic damping)
        self.lmbda_sharp = 1.5 # Phase discrimination sharpness
        self.R_th = 0.1     # Resonance threshold for T
        self.k_theta = 0.5  # Maturity coupling decay constant
        
        # --- Memory Optimization ---
        self.prev_S_ext = None # 초기 스파이크 방지를 위해 None으로 시작 (v1.2 수정 반영)

    def _get_alpha(self):
        # alpha(T): 성숙도에 따른 구조적 효율 (Moderate T에서 피크 후 감소)
        return self.alpha_0 * (1 + self.T) / (1 + 0.1 * self.T**2)

    def _get_beta(self):
        # beta(T): 성숙도에 따른 붕괴 저항성 (T가 높을수록 Resilience 증가)
        return self.beta_0 / (1 + self.T + 0.1 * self.T**2)

    def step(self, S_ext, A_ext, dt=0.01):
        """
        S_ext: 외부 참조 신호 (Wave 동기화용)
        A_ext: 외부 에너지/충격 (Flow/Resonance용)
        """
        # 0. 초기 스텝 스파이크 방지 로직 (v1.2 보정)
        if self.prev_S_ext is None:
            self.prev_S_ext = S_ext
            
        # dS/dt 계산
        dS_dt = (S_ext - self.prev_S_ext) / dt
        self.prev_S_ext = S_ext

        # 1. 존재 방정식 (E)
        # E(t) = F(t) * sin(phi_W(t)) * R(t)
        E = self.F * np.sin(self.phi_W) * self.R

        # 2. 파동 동역학 (W) - S_ext의 변화율(dS_dt)에만 반응
        theta_T = 1.0 / (1 + self.k_theta * self.T)
        d_phi = self.omega_base + theta_T * dS_dt + 0.05 * np.tanh(self.R)
        self.phi_W += d_phi * dt

        # 3. 흐름 동역학 (F) - A_ext(외부 에너지)가 동력으로 작용
        dF = -self.kappa * self.F + self.delta * A_ext + self.eta * E - self.mu * self.F**3
        self.F = max(0, self.F + dF * dt)

        # 4. 공명 동역학 (R) - A_ext(외부 충격)가 붕괴항으로 작용
        dR = self._get_alpha() * self.F * np.tanh(self.lmbda_sharp * np.sin(self.phi_W)) \
             - self._get_beta() * abs(A_ext) * self.R
        self.R = max(0, self.R + dR * dt)

        # 5. 진리 축적 (T) - v1.2 Ceiling 로직
        dT = self.sigma * (max(0, self.R - self.R_th) / (1 + self.T)) - self.lmbda_decay * self.T
        self.T += dT * dt

        return {'E': E, 'F': self.F, 'W': np.sin(self.phi_W), 'R': self.R, 'T': self.T}

# --- 시뮬레이션 환경 구축 ---
def run_fwr_simulation():
    core_6 = {
        'eta': 1.8, 'alpha_0': 0.6, 'beta_0': 0.5, 
        'sigma': 0.1, 'decay': 0.005, 'omega': 3.14
    }

    engine = FWREngineV12(core_6)
    history = {'E': [], 'F': [], 'W': [], 'R': [], 'T': []}
    
    steps = 3000
    dt = 0.01
    t_axis = np.linspace(0, steps*dt, steps)

    for i in range(steps):
        s_val = 1.0 + 0.5 * np.sin(0.5 * i * dt)
        a_val = 2.5 if 500 < i < 1000 else 0.5
        
        state = engine.step(s_val, a_val, dt)
        for k in history: history[k].append(state[k])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(t_axis, history['E'], label='Existence (E)', color='purple', linewidth=1.5)
    axes[0].fill_between(t_axis, history['W'], color='cyan', alpha=0.1, label='Wave Surface')
    axes[0].set_title("FWR v1.2: Existence Field & Wave Modulation")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, history['F'], label='Flow (F)', color='green')
    axes[1].plot(t_axis, history['R'], label='Resonance (R)', color='orange')
    axes[1].set_title("FWR v1.2: Energy Flow & Relational Resonance")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_axis, history['T'], label='Truth (T)', color='red', linewidth=2)
    axes[2].set_title("FWR v1.2: Structural Truth Accumulation (The Ceiling Effect)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fwr_simulation()
