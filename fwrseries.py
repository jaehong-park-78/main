import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # FWR 클러스터링을 위한 K-Means
from sklearn.metrics import silhouette_score # 클러스터링 평가
import random # FWR 옵티마이저를 위한 간단한 난수 사용

# --- 1. 다변량 시계열 데이터 생성 ---
np.random.seed(42)
num_points = 500
time = np.linspace(0, 10, num_points)

# Series A: 기본 사인 파동 + 노이즈 + 추세
data_A = np.sin(time * 2) + np.random.randn(num_points) * 0.5 + time * 0.2
# Series B: Series A와 유사하지만 약간의 지연 및 다른 주파수 + 노이즈
data_B = np.sin(time * 2.2 + 0.5) + np.random.randn(num_points) * 0.6 + time * 0.15
# Series C: Series A와 반대 위상 (초기에는 약한 공명 예상)
data_C = np.sin(time * 2 + np.pi) + np.random.randn(num_points) * 0.4 + time * 0.25

df_multi = pd.DataFrame({'Time': time, 'Series_A': data_A, 'Series_B': data_B, 'Series_C': data_C})

# --- FWR Core Functions ---

def calculate_fwr_components(series_data, alpha_flow=0.1, beta_wave=0.5, sampling_rate=(1/(time[1]-time[0]))):
    """단일 시계열에 대한 Flow, Wave, Resonance를 계산합니다."""
    # Flow: 변화율에 K(t) 역할을 하는 alpha_flow 적용
    flow = series_data.diff().fillna(0) * alpha_flow
    
    # Wave: 특정 주파수 대역의 파동 성분 추출 (여기서는 고주파 통과 필터 사용)
    nyquist = 0.5 * sampling_rate
    # 예시: 0.5 Hz 이상의 주파수 성분만 Wave로 추출
    cutoff_freq = 0.5 / nyquist 
    b, a = butter(2, cutoff_freq, btype='high', analog=False)
    wave = lfilter(b, a, series_data) * beta_wave # A(t) 역할을 하는 beta_wave 적용

    # Inner Resonance: 개별 Flow와 Wave의 강도 곱
    resonance_inner = np.abs(flow) * np.abs(wave)
    
    return flow, wave, resonance_inner

def calculate_coupling_strength(flow1, wave1, flow2, wave2, window_size=50):
    """두 시계열 간의 Coupling Strength C_ij(t)를 계산합니다."""
    # 정규화를 위해 스케일러 사용 (이동 상관계수 계산 위함)
    scaler = StandardScaler()

    # Flow와 Wave의 상관관계 (이동 상관계수)를 Coupling Strength로 간주
    # 각 시점의 Flow와 Wave를 결합한 벡터의 상관관계로 볼 수도 있으나 여기서는 단순화
    
    # 흐름과 파동의 정규화된 시계열 생성
    flow1_scaled = scaler.fit_transform(flow1.values.reshape(-1, 1)).flatten()
    flow2_scaled = scaler.fit_transform(flow2.values.reshape(-1, 1)).flatten()
    wave1_scaled = scaler.fit_transform(wave1.values.reshape(-1, 1)).flatten()
    wave2_scaled = scaler.fit_transform(wave2.values.reshape(-1, 1)).flatten()

    # 각 쌍의 이동 상관계수 계산
    coupling_flow = pd.Series(flow1_scaled).rolling(window=window_size).corr(pd.Series(flow2_scaled)).fillna(0)
    coupling_wave = pd.Series(wave1_scaled).rolling(window=window_size).corr(pd.Series(wave2_scaled)).fillna(0)
    
    # Flow와 Wave Coupling의 가중 평균으로 최종 C_ij(t) 정의 (가중치는 조절 가능)
    coupling_strength = (coupling_flow * 0.6 + coupling_wave * 0.4) 
    
    return coupling_strength.fillna(0)

def detect_resonance_phase(resonance_value, prev_resonance_value=None):
    """6단계 공명 위상 탐지 (규칙 기반, 단순화된 버전)."""
    if prev_resonance_value is None: # 첫 값은 Preservation으로 가정
        return "Preservation"

    diff = resonance_value - prev_resonance_value

    # 임계값은 예시이며, 실제로는 FWR 옵티마이저를 통해 학습/조정될 수 있습니다.
    if resonance_value < 0.05: # 낮은 공명 강도
        if diff < -0.01:
            return "Dissolution"
        elif diff < -0.05:
            return "Annihilation"
        else:
            return "Preservation"

    elif 0.05 <= resonance_value < 0.2: # 중간 공명 강도
        if diff > 0.01:
            return "Generation"
        elif diff < -0.01:
            return "Separation"
        else:
            return "Preservation"

    else: # resonance_value >= 0.2: # 높은 공명 강도
        if diff > 0.01:
            return "Fusion"
        elif diff < -0.01:
            return "Separation"
        else:
            return "Fusion"

# --- 2. FWR 구성 요소 계산 (각 시리즈 및 Coupling) ---
sampling_rate = 1 / (time[1] - time[0]) # 샘플링 주파수 계산

# 초기 FWR 매개변수 (FWR Optimizer에 의해 조정될 수 있음)
initial_alpha_flow = 0.1
initial_beta_wave = 0.5
initial_coupling_window_size = 50

for col in ['Series_A', 'Series_B', 'Series_C']:
    df_multi[f'{col}_Flow'], df_multi[f'{col}_Wave'], df_multi[f'{col}_Resonance_Inner'] = \
        calculate_fwr_components(df_multi[col], initial_alpha_flow, initial_beta_wave, sampling_rate)

# 다변량 Coupling Strength 계산
df_multi['Coupling_AB'] = calculate_coupling_strength(
    df_multi['Series_A_Flow'], df_multi['Series_A_Wave'],
    df_multi['Series_B_Flow'], df_multi['Series_B_Wave'], initial_coupling_window_size
)
df_multi['Coupling_AC'] = calculate_coupling_strength(
    df_multi['Series_A_Flow'], df_multi['Series_A_Wave'],
    df_multi['Series_C_Flow'], df_multi['Series_C_Wave'], initial_coupling_window_size
)

# Total System Resonance: 내적 공명 + Coupling Strength의 절대값 (강도이므로)
df_multi['Total_System_Resonance'] = \
    df_multi['Series_A_Resonance_Inner'] + \
    df_multi['Series_B_Resonance_Inner'] + \
    df_multi['Series_C_Resonance_Inner'] + \
    np.abs(df_multi['Coupling_AB']) * 0.5 + \
    np.abs(df_multi['Coupling_AC']) * 0.5

# Resonance Phase 감지
df_multi['Resonance_Phase'] = None
for i in range(len(df_multi)):
    if i == 0:
        df_multi.loc[i, 'Resonance_Phase'] = detect_resonance_phase(df_multi.loc[i, 'Total_System_Resonance'])
    else:
        df_multi.loc[i, 'Resonance_Phase'] = detect_resonance_phase(
            df_multi.loc[i, 'Total_System_Resonance'],
            df_multi.loc[i-1, 'Total_System_Resonance']
        )

# --- 3. FWR 클러스터 (FWR Cluster) ---
# FWR 특징 벡터를 구성하여 클러스터링 수행
# 특징: 각 시점의 Flow, Wave (각 시리즈), Coupling Strength, Total Resonance
fwr_features = df_multi[[
    'Series_A_Flow', 'Series_A_Wave',
    'Series_B_Flow', 'Series_B_Wave',
    'Series_C_Flow', 'Series_C_Wave',
    'Coupling_AB', 'Coupling_AC',
    'Total_System_Resonance'
]].copy()

# FWR 특징 스케일링 (K-Means 등 거리 기반 알고리즘에 필수)
scaler_fwr = StandardScaler()
fwr_features_scaled = scaler_fwr.fit_transform(fwr_features)

# 최적의 클러스터 개수 (k) 찾기 (간단한 실루엣 스코어 사용)
# 실제 구현에서는 Elbow method, Gap statistic 등 더 정교한 방법 사용
max_k = 10
silhouette_scores = []
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(fwr_features_scaled)
    score = silhouette_score(fwr_features_scaled, clusters)
    silhouette_scores.append(score)

optimal_k = np.argmax(silhouette_scores) + 2 # +2는 k가 2부터 시작했기 때문
print(f"\nOptimal number of FWR Clusters (based on Silhouette Score): {optimal_k}")

# 최적의 k로 K-Means 클러스터링 수행
kmeans_fwr = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_multi['FWR_Cluster'] = kmeans_fwr.fit_predict(fwr_features_scaled)

# 각 클러스터의 평균 FWR 특징 분석
cluster_means = df_multi.groupby('FWR_Cluster')[fwr_features.columns].mean()
print("\n--- FWR Cluster Mean Features ---")
print(cluster_means)

# 각 클러스터에 대한 Resonance Phase 분포 분석 (클러스터의 의미 해석에 도움)
print("\n--- Resonance Phase Distribution by FWR Cluster ---")
print(pd.crosstab(df_multi['FWR_Cluster'], df_multi['Resonance_Phase'], normalize='index'))


# --- 4. FWR 옵티마이저 (FWR Optimizer) ---
# 목표: Total_System_Resonance의 평균을 최대화 (단순화된 목표 함수)
# 최적화 대상 매개변수: alpha_flow, beta_wave

def objective_function(alpha_flow, beta_wave, df_input, coupling_window_size, sampling_rate):
    """주어진 매개변수로 FWR을 계산하고 Total_System_Resonance의 평균을 반환."""
    df_temp = df_input.copy()
    for col in ['Series_A', 'Series_B', 'Series_C']:
        df_temp[f'{col}_Flow'], df_temp[f'{col}_Wave'], df_temp[f'{col}_Resonance_Inner'] = \
            calculate_fwr_components(df_temp[col], alpha_flow, beta_wave, sampling_rate)

    df_temp['Coupling_AB'] = calculate_coupling_strength(
        df_temp['Series_A_Flow'], df_temp['Series_A_Wave'],
        df_temp['Series_B_Flow'], df_temp['Series_B_Wave'], coupling_window_size
    )
    df_temp['Coupling_AC'] = calculate_coupling_strength(
        df_temp['Series_A_Flow'], df_temp['Series_A_Wave'],
        df_temp['Series_C_Flow'], df_temp['Series_C_Wave'], coupling_window_size
    )
    
    total_resonance = \
        df_temp['Series_A_Resonance_Inner'] + \
        df_temp['Series_B_Resonance_Inner'] + \
        df_temp['Series_C_Resonance_Inner'] + \
        np.abs(df_temp['Coupling_AB']) * 0.5 + \
        np.abs(df_temp['Coupling_AC']) * 0.5
    
    return total_resonance.mean()

# 간단한 그리드 서치 (실제로는 베이즈 최적화, 강화 학습 등 사용)
best_resonance = -np.inf
best_params = {}

# 탐색할 매개변수 범위
param_grid = {
    'alpha_flow': np.linspace(0.05, 0.2, 5),
    'beta_wave': np.linspace(0.3, 0.7, 5)
}

print("\n--- FWR Optimizer: Searching for optimal parameters ---")
for alpha_f in param_grid['alpha_flow']:
    for beta_w in param_grid['beta_wave']:
        current_resonance = objective_function(alpha_f, beta_w, df_multi, initial_coupling_window_size, sampling_rate)
        if current_resonance > best_resonance:
            best_resonance = current_resonance
            best_params = {'alpha_flow': alpha_f, 'beta_wave': beta_w}
        # print(f"Testing alpha_flow={alpha_f:.2f}, beta_wave={beta_w:.2f}, Avg_Resonance={current_resonance:.4f}")

print(f"\nOptimal Parameters: {best_params}")
print(f"Maximized Average Total System Resonance: {best_resonance:.4f}")

# 최적화된 매개변수로 다시 FWR 계산 (옵티마이저의 결과 적용)
optimized_alpha_flow = best_params['alpha_flow']
optimized_beta_wave = best_params['beta_wave']

for col in ['Series_A', 'Series_B', 'Series_C']:
    df_multi[f'{col}_Flow_Opt'], df_multi[f'{col}_Wave_Opt'], df_multi[f'{col}_Resonance_Inner_Opt'] = \
        calculate_fwr_components(df_multi[col], optimized_alpha_flow, optimized_beta_wave, sampling_rate)

df_multi['Coupling_AB_Opt'] = calculate_coupling_strength(
    df_multi['Series_A_Flow_Opt'], df_multi['Series_A_Wave_Opt'],
    df_multi['Series_B_Flow_Opt'], df_multi['Series_B_Wave_Opt'], initial_coupling_window_size
)
df_multi['Coupling_AC_Opt'] = calculate_coupling_strength(
    df_multi['Series_A_Flow_Opt'], df_multi['Series_A_Wave_Opt'],
    df_multi['Series_C_Flow_Opt'], df_multi['Series_C_Wave_Opt'], initial_coupling_window_size
)

df_multi['Total_System_Resonance_Opt'] = \
    df_multi['Series_A_Resonance_Inner_Opt'] + \
    df_multi['Series_B_Resonance_Inner_Opt'] + \
    df_multi['Series_C_Resonance_Inner_Opt'] + \
    np.abs(df_multi['Coupling_AB_Opt']) * 0.5 + \
    np.abs(df_multi['Coupling_AC_Opt']) * 0.5

# --- 5. 시각화 (업데이트된 부분 포함) ---

plt.figure(figsize=(18, 20))

# 5.1. 원본 다변량 데이터 시각화
plt.subplot(5, 1, 1)
plt.plot(df_multi['Time'], df_multi['Series_A'], label='Series A', alpha=0.7)
plt.plot(df_multi['Time'], df_multi['Series_B'], label='Series B', alpha=0.7)
plt.plot(df_multi['Time'], df_multi['Series_C'], label='Series C', alpha=0.7)
plt.title('1. Original Multi-variate Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 5.2. Coupling Strength 시각화
plt.subplot(5, 1, 2)
plt.plot(df_multi['Time'], df_multi['Coupling_AB'], label='Coupling (A <-> B)', color='orange', alpha=0.8)
plt.plot(df_multi['Time'], df_multi['Coupling_AC'], label='Coupling (A <-> C)', color='blue', linestyle='--', alpha=0.8)
plt.title('2. Inter-series Resonance: Coupling Strength ($C_{ij}(t)$)')
plt.xlabel('Time')
plt.ylabel('Coupling Strength')
plt.legend()
plt.grid(True)

# 5.3. 총 시스템 Resonance와 Resonance Phase 시각화
plt.subplot(5, 1, 3)
plt.plot(df_multi['Time'], df_multi['Total_System_Resonance'], label='Total System Resonance', color='purple', linewidth=2)
phase_colors = {
    "Preservation": "gray", "Fusion": "red", "Generation": "green",
    "Separation": "blue", "Dissolution": "orange", "Annihilation": "black"
}
for phase, color in phase_colors.items():
    phase_times = df_multi[df_multi['Resonance_Phase'] == phase]['Time']
    phase_resonance = df_multi[df_multi['Resonance_Phase'] == phase]['Total_System_Resonance']
    plt.scatter(phase_times, phase_resonance, color=color, label=phase, s=10, alpha=0.6)
plt.title('3. Total System Resonance and Detected Resonance Phases')
plt.xlabel('Time')
plt.ylabel('Resonance Strength')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# 5.4. FWR 클러스터 시각화
plt.subplot(5, 1, 4)
scatter = plt.scatter(df_multi['Time'], df_multi['Total_System_Resonance'], 
                      c=df_multi['FWR_Cluster'], cmap='viridis', s=20, alpha=0.7)
plt.title('4. FWR Clusters over Time (Colored by Cluster ID)')
plt.xlabel('Time')
plt.ylabel('Total System Resonance')
plt.colorbar(scatter, label='FWR Cluster ID')
plt.grid(True)

# 5.5. 최적화 전후 Total System Resonance 비교
plt.subplot(5, 1, 5)
plt.plot(df_multi['Time'], df_multi['Total_System_Resonance'], label='Original Total Resonance', color='blue', alpha=0.7)
plt.plot(df_multi['Time'], df_multi['Total_System_Resonance_Opt'], label='Optimized Total Resonance', color='red', linestyle='--', alpha=0.8)
plt.title('5. Total System Resonance: Original vs. Optimized')
plt.xlabel('Time')
plt.ylabel('Resonance Strength')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# --- 결과 요약 ---
print("\n--- FWR Component Statistics (Sample) ---")
print(df_multi[['Series_A_Flow', 'Series_A_Wave', 'Series_A_Resonance_Inner', 
                'Coupling_AB', 'Coupling_AC', 'Total_System_Resonance']].describe())

print("\n--- Detected Resonance Phases Distribution ---")
print(df_multi['Resonance_Phase'].value_counts())

print("\n--- Optimal Parameters Found by FWR Optimizer ---")
print(f"Alpha Flow: {optimized_alpha_flow:.2f}")
print(f"Beta Wave: {optimized_beta_wave:.2f}")
print(f"Original Average Total Resonance: {df_multi['Total_System_Resonance'].mean():.4f}")
print(f"Optimized Average Total Resonance: {df_multi['Total_System_Resonance_Opt'].mean():.4f}")

print("\n--- Sample Data with FWR Components, Cluster, and Phase ---")
print(df_multi[['Time', 'Series_A', 'Series_B', 'Series_C', 
                'Total_System_Resonance', 'FWR_Cluster', 'Resonance_Phase', 
                'Total_System_Resonance_Opt']].head(10))
