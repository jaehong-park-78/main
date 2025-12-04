import numpy as np

# ==========================================================
# 1. FWR-Opt 최적화기 클래스 정의 (재사용)
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
# 2. 유틸리티 함수 (활성화, 손실)
# ==========================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    # y는 sigmoid(x)의 출력값
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    # ReLU의 입력값 x가 필요함
    return (x > 0).astype(x.dtype)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def initialize_fcn_params(input_size, hidden_size1, hidden_size2, output_size):
    np.random.seed(42)
    params = {}
    
    # Layer 1 (Input -> Hidden1)
    params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
    params['b1'] = np.zeros((1, hidden_size1))
    
    # Layer 2 (Hidden1 -> Hidden2)
    params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
    params['b2'] = np.zeros((1, hidden_size2))
    
    # Layer 3 (Hidden2 -> Output) - Softmax/Sigmoid 출력을 위해 Xavier 초기화 대신 조금 더 작게
    params['W3'] = np.random.randn(hidden_size2, output_size) * np.sqrt(1.0 / hidden_size2)
    params['b3'] = np.zeros((1, output_size))
    
    return params

def create_synthetic_fcn_data(N_samples, input_size):
    np.random.seed(42)
    # 5차원 특성 데이터 생성
    X_data = np.random.randn(N_samples, input_size)
    
    # 가상의 이진 분류 레이블 생성: 첫 2개 특성의 합이 0보다 크면 1, 아니면 0
    Y_data = (X_data[:, 0] + X_data[:, 1] > 0).astype(int)
    return X_data, Y_data.reshape(-1, 1)

# ==========================================================
# 3. FCN 순전파/역전파 구현
# ==========================================================

def fcn_forward(X, params):
    
    # Layer 1: Relu
    Z1 = X @ params['W1'] + params['b1']
    A1 = relu(Z1)
    
    # Layer 2: Relu
    Z2 = A1 @ params['W2'] + params['b2']
    A2 = relu(Z2)
    
    # Layer 3: Sigmoid (이진 분류)
    Z3 = A2 @ params['W3'] + params['b3']
    A3 = sigmoid(Z3)
    
    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache

def fcn_backward(Y, cache, params):
    N = Y.shape[0]
    
    # 1. 출력층 (Layer 3) 역전파 (Sigmoid + Binary Cross-Entropy)
    # dL/dZ3 = A3 - Y (Binary Cross-Entropy Loss의 간단한 기울기)
    dZ3 = (cache['A3'] - Y) / N
    
    grads = {}
    
    # dL/dW3 = A2^T * dZ3
    grads['W3'] = cache['A2'].T @ dZ3
    # dL/db3
    grads['b3'] = np.sum(dZ3, axis=0, keepdims=True)
    
    # 2. Layer 2 역전파 (ReLU)
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * drelu(cache['Z2'])
    
    # dL/dW2 = A1^T * dZ2
    grads['W2'] = cache['A1'].T @ dZ2
    # dL/db2
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True)
    
    # 3. Layer 1 역전파 (ReLU)
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * drelu(cache['Z1'])
    
    # dL/dW1 = X^T * dZ1
    grads['W1'] = cache['X'].T @ dZ1
    # dL/db1
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True)
    
    return grads

def calculate_binary_cross_entropy(Y_batch, Y_pred):
    epsilon = 1e-8
    loss = -np.mean(Y_batch * np.log(Y_pred + epsilon) + (1 - Y_batch) * np.log(1 - Y_pred + epsilon))
    return loss

# ==========================================================
# 4. 훈련 설정 및 실행
# ==========================================================

# 하이퍼파라미터
N_SAMPLES = 2000     
INPUT_SIZE = 5       
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 1      
EPOCHS = 100         
LR = 0.005           
BATCH_SIZE = 64      

# 데이터 생성
X_data, Y_data = create_synthetic_fcn_data(N_SAMPLES, INPUT_SIZE)

# 매개변수 초기화
params = initialize_fcn_params(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
# FWR-Opt 초기화
fwr_opt = FWROptimizer(lr=LR)

print(f"--- FWR-Opt를 사용한 FCN 이진 분류 모델 훈련 시작 ---")
print(f"샘플 수: {N_SAMPLES}, 은닉층1: {HIDDEN_SIZE1}, 은닉층2: {HIDDEN_SIZE2}, LR: {LR}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

# 훈련 루프
for epoch in range(EPOCHS):
    permutation = np.random.permutation(N_SAMPLES)
    X_shuffled = X_data[permutation]
    Y_shuffled = Y_data[permutation]
    
    epoch_losses = []
    correct_predictions = 0
    total_samples = 0
    
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        X_batch = X_shuffled[i:i + BATCH_SIZE]
        Y_batch = Y_shuffled[i:i + BATCH_SIZE]
        
        # 1. 순전파 실행
        Y_pred, cache = fcn_forward(X_batch, params)
        loss = calculate_binary_cross_entropy(Y_batch, Y_pred)
            
        # 2. 역전파 실행
        grads = fcn_backward(Y_batch, cache, params)

        epoch_losses.append(loss)
        
        # 정확도 계산
        Y_pred_labels = (Y_pred > 0.5).astype(int)
        correct_predictions += np.sum(Y_pred_labels == Y_batch)
        total_samples += Y_batch.shape[0]

        # 3. FWR-Opt를 사용하여 매개변수 업데이트
        params = fwr_opt.update(params, grads)
    
    if epoch_losses: 
        avg_loss = np.mean(epoch_losses)
        epoch_accuracy = (correct_predictions / total_samples) * 100

        # 손실 출력 및 정확도 계산
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d} | Average Loss: {avg_loss:.6f} | Accuracy: {epoch_accuracy:.2f}%")
        
print("\n--- 훈련 완료 ---")

# ==========================================================
# 5. 최종 예측 결과 확인 (전체 데이터셋)
# ==========================================================
Y_pred_final, _ = fcn_forward(X_data, params)
final_loss = calculate_binary_cross_entropy(Y_data, Y_pred_final)

Y_pred_labels = (Y_pred_final > 0.5).astype(int)
final_accuracy = np.mean(Y_pred_labels == Y_data) * 100

print(f"최종 평균 Binary Cross-Entropy Loss: {final_loss:.6f}")
print(f"최종 훈련 정확도: {final_accuracy:.2f}%")
print("\nFWR-Opt는 FCN 모델을 사용하여 이진 분류를 성공적으로 수행했습니다.")
