import numpy as np
import time

# ==========================================================
# 1. FWR-Opt 최적화기 클래스 정의 (MNIST와 동일)
# ==========================================================
class FWROptimizer:
    """
    FWR-Opt (Flow-Wave-Resonance Optimizer)
    """
    def __init__(self, lr=0.001, beta_F=0.9, beta_W=0.999, epsilon=1e-8, gamma_W=0.01, gamma_R=0.9):
        self.lr = lr
        self.beta_F = beta_F
        self.v = {}         # Flow (모멘텀)
        self.beta_W = beta_W
        self.s = {}         # Wave (2차 모멘텀)
        self.gamma_W = gamma_W 
        self.epsilon = epsilon  
        self.gamma_R = gamma_R 
        self.r_coherence = {} # Resonance (일관성)
        self.t = 0 

    def update(self, params, grads):
        self.t += 1

        for key in params.keys():
            g = grads[key]
            
            if key not in self.v:
                self.v[key] = np.zeros_like(g)
                self.s[key] = np.zeros_like(g)
                self.r_coherence[key] = np.ones_like(g)

            # F-Component (Flow): 모멘텀 업데이트
            self.v[key] = self.beta_F * self.v[key] + (1 - self.beta_F) * g
            v_hat = self.v[key] / (1 - self.beta_F**self.t)
            
            # W-Component (Wave): 적응성 및 안정성 업데이트
            self.s[key] = self.beta_W * self.s[key] + (1 - self.beta_W) * (g**2)
            s_hat = self.s[key] / (1 - self.beta_W**self.t)
            
            # Wave Dampening 적용
            wave_adjusted_std = np.sqrt(s_hat) + self.epsilon + self.gamma_W

            # R-Component (Resonance): 공명 및 일관성 업데이트
            coherence_mask = np.sign(g) * np.sign(v_hat)
            
            # Coherence가 높을 경우 1.0, 낮을 경우 감쇠
            self.r_coherence[key] = np.where(
                coherence_mask > 0,
                1.0, 
                self.gamma_R * self.r_coherence[key]
            )
            
            R_factor = 1 + self.r_coherence[key]

            # 최종 업데이트 계산 (E = F * W * R)
            update_step = (self.lr * v_hat) / wave_adjusted_std * R_factor

            # 매개변수 업데이트
            params[key] -= update_step

        return params
# ==========================================================


# ==========================================================
# 2. MLP 유틸리티 함수
# ==========================================================

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    """안정적인 softmax 계산"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(Y_batch, Y_pred):
    """Categorical Cross-Entropy 손실"""
    N = Y_batch.shape[0]
    Y_pred = np.clip(Y_pred, 1e-12, 1. - 1e-12) 
    loss = -np.sum(Y_batch * np.log(Y_pred)) / N
    return loss

def one_hot_encode(Y, num_classes):
    """정수 레이블을 원-핫 인코딩으로 변환"""
    N = Y.shape[0]
    Y_oh = np.zeros((N, num_classes))
    if Y.ndim > 1:
        Y = Y.flatten()
    Y_oh[np.arange(N), Y] = 1
    return Y_oh

def create_synthetic_cifar100_data(N_samples, input_size, num_classes):
    """
    CIFAR-100과 유사한 가상 데이터셋 생성 (고차원, 다중 클래스)
    """
    np.random.seed(42)
    # 데이터: (N, 3072), 0~1 사이의 값으로 정규화
    X_data = np.random.rand(N_samples, input_size) * 0.7
    
    # 레이블: 0부터 num_classes-1까지 랜덤 정수 레이블
    Y_labels = np.random.randint(0, num_classes, N_samples).reshape(-1, 1)
    
    # 레이블 기반으로 데이터에 더 복잡한 패턴 추가
    for i in range(N_samples):
        label = Y_labels[i, 0]
        # 클래스 ID에 따라 특징 공간의 다른 영역에 더 강한 신호를 부여
        pattern_start = (label * 30) % input_size 
        pattern_end = (pattern_start + 100) % input_size 
        
        # 순환 버퍼처럼 인덱스 처리
        if pattern_start < pattern_end:
            X_data[i, pattern_start:pattern_end] += np.random.rand(pattern_end - pattern_start) * 0.5
        else: # 패턴이 끝에서 시작으로 넘어가는 경우
            X_data[i, pattern_start:] += np.random.rand(input_size - pattern_start) * 0.5
            X_data[i, :pattern_end] += np.random.rand(pattern_end) * 0.5
    
    # 데이터 정규화 [0, 1]
    X_data = np.clip(X_data, 0, 1)
    
    return X_data, Y_labels

# ==========================================================
# 3. 3-Layer MLP 모델 순전파/역전파
# ==========================================================

def initialize_mlp_params(input_size, hidden_size_1, hidden_size_2, output_size):
    """3-Layer MLP 가중치 초기화 (He 초기화 사용)"""
    np.random.seed(42)
    params = {}
    
    # Layer 1 (Input -> Hidden 1)
    params['W1'] = np.random.randn(input_size, hidden_size_1) * np.sqrt(2. / input_size)
    params['b1'] = np.zeros((1, hidden_size_1))
    
    # Layer 2 (Hidden 1 -> Hidden 2)
    params['W2'] = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2. / hidden_size_1)
    params['b2'] = np.zeros((1, hidden_size_2))

    # Layer 3 (Hidden 2 -> Output)
    params['W3'] = np.random.randn(hidden_size_2, output_size) * np.sqrt(1. / hidden_size_2)
    params['b3'] = np.zeros((1, output_size))
    
    return params

def mlp_forward(X, params):
    """순전파: 3계층 (ReLU, ReLU, Softmax)"""
    
    # Layer 1 (Input -> H1)
    Z1 = X @ params['W1'] + params['b1']
    A1 = relu(Z1)
    
    # Layer 2 (H1 -> H2)
    Z2 = A1 @ params['W2'] + params['b2']
    A2 = relu(Z2)

    # Layer 3 (H2 -> Output)
    Z3 = A2 @ params['W3'] + params['b3']
    Y_pred = softmax(Z3)
    
    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'Y_pred': Y_pred}
    return Y_pred, cache

def mlp_backward(Y_true_oh, cache, params):
    """역전파: 3계층"""
    N = Y_true_oh.shape[0]
    grads = {}
    
    # 1. 출력층 기울기 (dL/dZ3)
    dZ3 = (cache['Y_pred'] - Y_true_oh) / N # (N, Output_Size)
    
    # dL/dW3: A2^T @ dZ3
    grads['W3'] = cache['A2'].T @ dZ3
    # dL/db3: sum(dZ3)
    grads['b3'] = np.sum(dZ3, axis=0, keepdims=True)
    
    # 2. Layer 2 역전파 시작 (dL/dA2)
    dA2 = dZ3 @ params['W3'].T # (N, H2_Size)
    dZ2 = dA2 * drelu(cache['Z2']) # (N, H2_Size)

    # dL/dW2: A1^T @ dZ2
    grads['W2'] = cache['A1'].T @ dZ2
    # dL/db2: sum(dZ2)
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)
    
    # 3. Layer 1 역전파 시작 (dL/dA1)
    dA1 = dZ2 @ params['W2'].T # (N, H1_Size)
    dZ1 = dA1 * drelu(cache['Z1']) # (N, H1_Size)
    
    # dL/dW1: X^T @ dZ1
    grads['W1'] = cache['X'].T @ dZ1
    # dL/db1: sum(dZ1)
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)
    
    return grads

# ==========================================================
# 4. 훈련 설정 및 실행
# ==========================================================

# 하이퍼파라미터 (CIFAR-100 시뮬레이션을 위한 조정)
INPUT_SIZE = 32 * 32 * 3 # 3072
HIDDEN_SIZE_1 = 512    
HIDDEN_SIZE_2 = 256
OUTPUT_SIZE = 100        # 100 classes
N_SAMPLES = 10000        # 더 많은 샘플
EPOCHS = 100             # 더 많은 Epoch 필요
LR = 0.001               # 복잡도가 높아져 학습률을 약간 낮춤
BATCH_SIZE = 128         # 배치 크기 증가

# 데이터 생성 및 준비
X_data, Y_labels = create_synthetic_cifar100_data(N_SAMPLES, INPUT_SIZE, OUTPUT_SIZE)
Y_data_oh = one_hot_encode(Y_labels, OUTPUT_SIZE)

# 매개변수 초기화
params = initialize_mlp_params(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
# FWR-Opt 초기화
fwr_opt = FWROptimizer(lr=LR, gamma_R=0.9, beta_F=0.9) 

print(f"--- FWR-Opt를 사용한 가상 CIFAR-100 MLP 훈련 시작 (3계층) ---")
print(f"모델 구조: {INPUT_SIZE} -> {HIDDEN_SIZE_1} -> {HIDDEN_SIZE_2} -> {OUTPUT_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
start_time = time.time()

# 훈련 루프
for epoch in range(EPOCHS):
    permutation = np.random.permutation(N_SAMPLES)
    X_shuffled = X_data[permutation]
    Y_oh_shuffled = Y_data_oh[permutation]
    Y_labels_shuffled = Y_labels[permutation]
    
    epoch_losses = []
    correct_predictions = 0
    total_samples = 0
    
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        X_batch = X_shuffled[i:i + BATCH_SIZE]
        Y_oh_batch = Y_oh_shuffled[i:i + BATCH_SIZE]
        Y_labels_batch = Y_labels_shuffled[i:i + BATCH_SIZE] 
        
        # 1. 순전파 실행
        Y_pred, cache = mlp_forward(X_batch, params)
        loss = cross_entropy_loss(Y_oh_batch, Y_pred)
            
        # 2. 역전파 실행
        grads = mlp_backward(Y_oh_batch, cache, params)

        epoch_losses.append(loss)
        
        # 정확도 계산
        Y_pred_labels_1d = np.argmax(Y_pred, axis=1) 
        Y_true_labels_1d = Y_labels_batch.flatten() 
        
        correct_predictions += np.sum(Y_pred_labels_1d == Y_true_labels_1d)
        total_samples += X_batch.shape[0]

        # 3. FWR-Opt를 사용하여 매개변수 업데이트
        params = fwr_opt.update(params, grads)
    
    if total_samples > 0: 
        avg_loss = np.mean(epoch_losses)
        epoch_accuracy = (correct_predictions / total_samples) * 100

        # 손실 출력 및 정확도 계산
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:2d} | Avg Loss: {avg_loss:.6f} | Accuracy: {epoch_accuracy:.2f}%")
        
end_time = time.time()
print(f"\n--- 훈련 완료 (총 시간: {end_time - start_time:.2f}초) ---")

# ==========================================================
# 5. 최종 예측 결과 확인
# ==========================================================
Y_pred_final, _ = mlp_forward(X_data, params)
final_loss = cross_entropy_loss(Y_data_oh, Y_pred_final)

Y_pred_labels = np.argmax(Y_pred_final, axis=1).reshape(-1, 1)
final_accuracy = np.mean(Y_pred_labels == Y_labels) * 100

print(f"\n최종 전체 데이터셋 Cross-Entropy Loss: {final_loss:.6f}")
print(f"최종 훈련 정확도: {final_accuracy:.2f}%")
print("\nFWR-Opt는 가상 CIFAR-100 유사 데이터셋의 3계층 MLP 분류 가중치 최적화에 사용되었습니다.")
