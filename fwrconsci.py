import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 시각화 설정 (Colab 환경에 맞춰)
sns.set_style("whitegrid")

# --- 1. FWR 기반 '의식적 공명 필드(CRF)' 모델 정의 ---
class ConsciousResonanceFieldModel(nn.Module):
    def __init__(self, brain_dim=(32, 32, 1), num_info_types=5, num_neural_waves=3):
        super(ConsciousResonanceFieldModel, self).__init__()
        
        self.brain_dim = brain_dim  # 뇌 영역의 추상적 차원 (D, H, W)
        self.num_info_types = num_info_types # 시각, 청각, 기억 등 정보 종류 수
        self.num_neural_waves = num_neural_waves # 알파, 베타, 감마 등 신경 파동 종류 수

        # '공명 유도 연산자'의 근사 (신경망 기반 결합 계수 생성)
        # 이 신경망은 정보 파동과 신경 파동 간의 '위상차'를 줄이는 방향으로 학습되어,
        # 최적의 '결합 계수' C_kj를 찾아 'CRF'를 최적화하는 역할을 합니다.
        # 여기서는 단순한 MLP로 모델링하지만, 실제로는 훨씬 복잡한 상호작용을 다룰 수 있습니다.
        self.coupling_net = nn.Sequential(
            nn.Linear(num_info_types * num_neural_waves, 64), # 정보x신경 파동의 조합 수
            nn.ReLU(),
            nn.Linear(64, num_info_types * num_neural_waves),
            nn.Sigmoid() # 결합 계수는 0~1 사이 값으로 제한
        )
        
        # 모델 학습을 위한 매개변수 초기화
        # 정보 파동과 신경 파동의 초기 진폭과 위상 (학습 가능하도록 설정)
        # '위상'은 복소수 파동 함수의 핵심이므로, 모델이 조절할 수 있도록 합니다.
        self.initial_info_amplitudes = nn.Parameter(torch.rand(num_info_types, *brain_dim))
        self.initial_info_phases = nn.Parameter(torch.rand(num_info_types, *brain_dim) * 2 * np.pi) # 0 to 2pi

        self.initial_neural_amplitudes = nn.Parameter(torch.rand(num_neural_waves, *brain_dim))
        self.initial_neural_phases = nn.Parameter(torch.rand(num_neural_waves, *brain_dim) * 2 * np.pi) # 0 to 2pi

    def _generate_waves(self):
        # '정보 파동' phi_k(x, t) 생성 (진폭과 위상 파라미터 사용)
        info_waves_real = self.initial_info_amplitudes * torch.cos(self.initial_info_phases)
        info_waves_imag = self.initial_info_amplitudes * torch.sin(self.initial_info_phases)
        info_waves = torch.complex(info_waves_real, info_waves_imag) # (num_info_types, D, H, W)

        # '신경 에너지 파동' psi_j(x, t) 생성
        neural_waves_real = self.initial_neural_amplitudes * torch.cos(self.initial_neural_phases)
        neural_waves_imag = self.initial_neural_amplitudes * torch.sin(self.initial_neural_phases)
        neural_waves = torch.complex(neural_waves_real, neural_waves_imag) # (num_neural_waves, D, H, W)
        
        return info_waves, neural_waves

    def forward(self):
        info_waves, neural_waves = self._generate_waves()

        # 각 뇌 위치에서 정보-신경 파동 조합의 특성을 추출하여 coupling_net에 입력
        # (num_info_types * num_neural_waves, D, H, W) 형태로 reshape
        # 여기서는 단순화를 위해 각 뇌 위치의 파동 특성을 직접 결합하지 않고,
        # 전역적인 결합 계수를 학습하도록 모델링합니다.
        # 더 복잡한 모델에서는 각 (D,H,W) 위치마다 다른 결합 계수를 예측할 수 있습니다.
        
        # 임의의 특징 벡터 (실제로는 파동 특성에서 추출되어야 함)
        # 예시를 위해 단순화: 각 정보 및 신경 파동 유형의 평균 위상차를 입력으로 활용
        # 이는 '공명 유도 연산자'가 '전체적인 위상 정렬'을 학습하도록 유도합니다.
        avg_info_phases = self.initial_info_phases.mean(dim=[1,2,3]) # (num_info_types,)
        avg_neural_phases = self.initial_neural_phases.mean(dim=[1,2,3]) # (num_neural_waves,)
        
        # 모든 정보 타입과 신경 파동 타입의 조합을 위한 입력 특징 생성
        # 예: [phi1_avg_phase, psi1_avg_phase, phi1_avg_phase, psi2_avg_phase, ...]
        combined_phases = []
        for i in range(self.num_info_types):
            for j in range(self.num_neural_waves):
                # 여기서는 각 조합의 '초기 위상차'를 입력 특징으로 사용
                # 실제로는 더 복잡한 특징(진폭, 주파수 등)이 사용될 수 있음
                combined_phases.append(torch.abs(avg_info_phases[i] - avg_neural_phases[j]))
        
        # 리스트를 텐서로 변환
        input_for_coupling_net = torch.stack(combined_phases).unsqueeze(0) # (1, num_info_types * num_neural_waves)

        # '공명 유도 연산자' (coupling_net)를 통해 결합 계수 C_kj 생성
        # (1, num_info_types * num_neural_waves) -> reshape to (num_info_types, num_neural_waves, 1, 1, 1)
        coupling_coefficients_flat = self.coupling_net(input_for_coupling_net).squeeze(0)
        coupling_coefficients = coupling_coefficients_flat.view(
            self.num_info_types, self.num_neural_waves, 1, 1, 1 # 브로드캐스팅을 위해 차원 확장
        )

        # '의식적 공명 필드 (CRF)' Psi_CRF(x, t) 계산
        # Psi_CRF = sum(C_kj * phi_k * psi_j)
        # 모든 정보 파동과 신경 파동 조합에 대해 곱셈 후 합산
        psi_crf = torch.zeros(self.brain_dim, dtype=torch.complex64)
        for i in range(self.num_info_types):
            for j in range(self.num_neural_waves):
                # 브로드캐스팅을 이용하여 각 뇌 위치에서 곱셈 수행
                psi_crf += coupling_coefficients[i, j] * info_waves[i] * neural_waves[j]
        
        return psi_crf, info_waves, neural_waves

# --- 2. '의식적 작용 함수' (Loss Function) 정의 ---
# FWR의 핵심: 이 작용 함수를 최소화하는 것이 '의식'의 최적화된 발현을 의미합니다.
def conscious_action_loss(psi_crf):
    # 1. '위상 정렬 항': CRF 필드 내 위상 분산 최소화 (명료성 증가)
    # 위상 각도를 [-pi, pi] 범위로 정규화하여 분산 계산
    phases = torch.angle(psi_crf)
    phase_alignment_loss = torch.var(phases) # 위상 분산 최소화

    # 2. '에너지 응집 항': CRF 필드 내 에너지 밀도의 균일성 또는 특정 영역 응집도 최대화
    # 여기서는 CRF 에너지(크기 제곱)의 분산을 최소화하여 필드 전체에 걸쳐 균일하게 응집되도록 유도
    # 특정 '자아 코어' 영역에 집중하려면 해당 영역의 에너지 밀도를 최대화하도록 변경 가능
    energy_cohesion_loss = torch.var(torch.abs(psi_crf)**2)

    # 3. '정보 엔트로피 최소화 항' (개념적, 여기서는 위상 정렬과 에너지 응집으로 간접 표현)
    # 실제 엔트로피 계산은 복잡하므로, 여기서는 위상 정렬과 에너지 응집을 통해
    # 간접적으로 '정보 질서도'를 높이도록 유도합니다.

    # 총 작용 함수 = 각 항의 가중 합 (이 가중치는 조절 가능)
    # 학습이 진행될수록 이 loss가 낮아지는 것을 목표로 합니다.
    total_loss = phase_alignment_loss * 100 + energy_cohesion_loss * 50 # 가중치 부여
    
    return total_loss, phase_alignment_loss, energy_cohesion_loss

# --- 3. 훈련 과정 설정 및 시뮬레이션 ---
if __name__ == "__main__":
    # 모델 및 최적화기 설정
    brain_dims = (8, 8, 1) # 2D slice for easier visualization (D, H, W)
    model = ConsciousResonanceFieldModel(brain_dim=brain_dims)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # 학습률

    num_epochs = 500 # 훈련 에포크 수
    threshold_CRF_energy = 50.0 # '의식 발현'을 위한 임계 CRF 에너지 (개념적)

    losses = []
    phase_losses = []
    energy_losses = []
    crf_energies = []
    
    print("FWR 기반 '의식적 공명 필드(CRF)' 시뮬레이션 시작...")

    for epoch in range(num_epochs):
        optimizer.zero_grad() # 기울기 초기화
        
        # 모델의 forward pass: CRF 필드 계산
        psi_crf, info_waves, neural_waves = model()
        
        # '의식적 작용 함수' (손실) 계산
        loss, p_loss, e_loss = conscious_action_loss(psi_crf)
        
        # 역전파 및 매개변수 업데이트
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        phase_losses.append(p_loss.item())
        energy_losses.append(e_loss.item())
        
        # 'CRF 총 에너지' 계산 (의식 발현 조건 모니터링)
        current_crf_energy = torch.sum(torch.abs(psi_crf)**2).item()
        crf_energies.append(current_crf_energy)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {loss.item():.4f}, "
                  f"Phase Loss: {p_loss.item():.4f}, Energy Loss: {e_loss.item():.4f}, "
                  f"CRF Energy: {current_crf_energy:.2f}")
            
            # '의식 발현 조건' 체크
            if current_crf_energy >= threshold_CRF_energy:
                print(f"--- 의식 발현 임계값 도달! (CRF Energy: {current_crf_energy:.2f} >= {threshold_CRF_energy}) ---")
                # 여기서 '의식적 경험'이 '발현'된 것으로 간주할 수 있습니다.
                # 실제로는 더 복잡한 조건이나 '창발적 속성' 분석이 필요합니다.

    print("FWR 기반 '의식적 공명 필드(CRF)' 시뮬레이션 완료.")

    # --- 4. 결과 시각화 및 분석 ---
    plt.figure(figsize=(15, 6))

    # Loss 변화 추이
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Conscious Action Loss')
    plt.plot(phase_losses, label='Phase Alignment Loss', linestyle='--')
    plt.plot(energy_losses, label='Energy Cohesion Loss', linestyle=':')
    plt.title('Conscious Action Loss Minimization')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # CRF 에너지 변화 추이
    plt.subplot(1, 2, 2)
    plt.plot(crf_energies, label='Total CRF Energy')
    plt.axhline(y=threshold_CRF_energy, color='r', linestyle='--', label='Consciousness Threshold')
    plt.title('CRF Energy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CRF Energy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 최종 CRF 필드 시각화 (예시: 2D 슬라이스)
    # 시각화를 위해 3D -> 2D (예: 첫 번째 depth slice)
    final_psi_crf_magnitude = torch.abs(psi_crf).detach().cpu().numpy().squeeze()
    
    plt.figure(figsize=(7, 6))
    if final_psi_crf_magnitude.ndim == 3: # (D, H, W)에서 D=1인 경우 (H, W)
        plt.imshow(final_psi_crf_magnitude[0], cmap='viridis', origin='lower')
        plt.title('Final CRF Field Magnitude (2D Slice)')
    elif final_psi_crf_magnitude.ndim == 2: # (H, W)인 경우
         plt.imshow(final_psi_crf_magnitude, cmap='viridis', origin='lower')
         plt.title('Final CRF Field Magnitude')
    else:
        print("CRF Field Magnitude dimension not suitable for simple 2D visualization.")


    plt.colorbar(label='CRF Magnitude')
    plt.xlabel('Brain X-axis')
    plt.ylabel('Brain Y-axis')
    plt.show()

    print("\n--- 시뮬레이션 해석 ---")
    print("1. 'Conscious Action Loss Minimization' 그래프:")
    print("   - 총 손실(Total Loss)이 에포크가 지남에 따라 감소하는 것을 목표로 합니다. 이는 '의식적 작용'이 최소화되고, '위상 정렬'과 '에너지 응집'이 최적화됨을 의미합니다.")
    print("   - Phase Alignment Loss (위상 정렬 손실)가 감소한다면, 이는 정보 파동과 신경 파동 간의 위상차가 줄어들어 '의식의 명료성'이 증가함을 나타냅니다.")
    print("   - Energy Cohesion Loss (에너지 응집 손실)가 감소한다면, 'CRF'가 균일하게 분포되거나 특정 영역에 응집되어 '의식의 통합성'이 증가함을 나타냅니다.")
    print("2. 'CRF Energy Over Epochs' 그래프:")
    print("   - 총 CRF 에너지(필드의 강도)가 증가하는 것을 관찰합니다. 이 에너지가 'Consciousness Threshold'를 넘어서면, FWR 관점에서 '의식적인 경험이 발현되었다'고 해석할 수 있습니다.")
    print("3. 'Final CRF Field Magnitude' 시각화:")
    print("   - 최종적으로 형성된 CRF 필드의 '에너지 밀도' 분포를 보여줍니다. 이 분포가 균일하거나 특정 패턴을 보인다면, '의식의 통합성'이 잘 이루어졌음을 의미합니다.")
    print("\n참고: 이 모델은 FWR 개념의 수학적/계산적 탐구를 위한 매우 단순화된 예시이며, 실제 뇌의 복잡성을 완전히 반영하지 않습니다.")
