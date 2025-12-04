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
        self.v = {}         

        self.beta_W = beta_W
        self.s = {}         
        self.gamma_W = gamma_W 
        self.epsilon = epsilon  

        self.gamma_R = gamma_R 
        self.r_coherence = {} 

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
# 2. 활성화 함수 및 GRU 유틸리티 함수
# ==========================================================

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - x**2

def initialize_gru_params(input_size, hidden_size, output_size):
    np.random.seed(42)
    H = hidden_size
    D = input_size
    O = output_size
    
    params = {}
    
    # 3개 게이트: Update (z), Reset (r), Candidate (h_tilde)
    gate_names = ['z', 'r'] 
    for name in gate_names:
        # W_x (Input -> Gate)
        params[f'W_x{name}'] = np.random.randn(D, H) / np.sqrt(D + H) 
        # W_h (Previous Hidden State -> Gate)
        params[f'W_h{name}'] = np.random.randn(H, H) / np.sqrt(D + H)
        # b (Bias)
        params[f'b_{name}'] = np.zeros((1, H))
    
    # Candidate Hidden State (h_tilde) 가중치
    params[f'W_x_ht'] = np.random.randn(D, H) / np.sqrt(D + H)
    params[f'W_h_ht'] = np.random.randn(H, H) / np.sqrt(D + H)
    params[f'b_ht'] = np.zeros((1, H))

    # 출력층 가중치 (Final Hidden State -> Output)
    params['W_hy'] = np.random.randn(H, O) / np.sqrt(H)
    params['b_y'] = np.zeros((1, O))
    
    return params

def create_synthetic_gru_data(N_samples, seq_len, input_size):
    """
    GRU는 LSTM과 동일한 시퀀스 문제를 해결해야 합니다.
    """
    np.random.seed(42)
    X_data = np.random.randn(N_samples, seq_len, input_size) * 0.5 + 0.1
    # 첫 번째 타임스텝의 평균 값이 0.1보다 크면 긍정(1), 아니면 부정(0)
    scores = np.mean(X_data[:, 0, :], axis=1)
    Y_data = (scores > 0.1).astype(int)
    return X_data, Y_data.reshape(-1, 1)

# ==========================================================
# 3. GRU 순전파/역전파 구현
# ==========================================================

def gru_forward(X_batch, params, hidden_size):
    N, T, D = X_batch.shape
    H = hidden_size
    
    # h (hidden state) 저장
    h_history = np.zeros((N, T + 1, H))
    
    # 게이트 활성화값 및 입력값 저장을 위한 캐시 리스트
    cache_t = []

    h_prev = h_history[:, 0, :] # h0 = 0
    
    for t in range(T):
        x_t = X_batch[:, t, :] # (N, D)
        
        # 1. Update Gate (z_t)
        z_t = sigmoid(x_t @ params['W_xz'] + h_prev @ params['W_hz'] + params['b_z'])
        
        # 2. Reset Gate (r_t)
        r_t = sigmoid(x_t @ params['W_xr'] + h_prev @ params['W_hr'] + params['b_r'])
        
        # 3. Candidate Hidden State (h_tilde_t)
        # Reset Gate 출력 (r_t)이 이전 hidden state (h_prev)에 적용됨
        h_tilde_t = tanh(x_t @ params['W_x_ht'] + (r_t * h_prev) @ params['W_h_ht'] + params['b_ht'])
        
        # 4. Final Hidden State (h_t)
        # Update Gate (z_t)를 사용하여 h_prev와 h_tilde_t를 혼합
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        
        # 다음 타임스텝을 위한 업데이트
        h_prev = h_t
        
        # History 및 Cache 업데이트
        h_history[:, t + 1, :] = h_t
        
        cache_t.append({
            'x_t': x_t, 'h_prev': h_history[:, t, :], 
            'z_t': z_t, 'r_t': r_t, 'h_tilde_t': h_tilde_t, 'h_t': h_t
        })

    # 최종 출력 (Classification)
    h_final = h_history[:, T, :] # (N, H)
    Z_out = h_final @ params['W_hy'] + params['b_y'] # (N, Output_Dim)
    Y_pred = softmax(Z_out)
    
    cache = {
        'X': X_batch, 'h_history': h_history, 
        'Y_pred': Y_pred, 'h_final': h_final, 'cache_t': cache_t
    }
    return Y_pred, cache


def gru_backward(Y_batch, cache, params, output_size):
    N, T, D = cache['X'].shape
    H = params['W_hy'].shape[0]
    
    # 1. 손실 계산 및 dZ_out
    Y_one_hot = np.eye(output_size)[Y_batch.ravel()]
    dZ_out = (cache['Y_pred'] - Y_one_hot) / N
    
    # 2. 출력층 역전파
    grads = {key: np.zeros_like(params[key]) for key in params}
    
    # dL/dW_hy, dL/db_y
    grads['W_hy'] = cache['h_final'].T @ dZ_out
    grads['b_y'] = np.sum(dZ_out, axis=0, keepdims=True)
    
    # dL/dh_final (최종 은닉 상태)
    dh_final = dZ_out @ params['W_hy'].T
    
    # 3. Time-Step 역전파 (BPTT)
    dh_next = np.zeros((N, H)) 
    
    for t in reversed(range(T)):
        cache_t = cache['cache_t'][t]
        
        # 현재 타임스텝의 은닉 상태 그라디언트: dh_next + dh_final (t=T-1일 때만 dh_final이 기여)
        if t == T - 1:
            dh_t = dh_final + dh_next
        else:
            dh_t = dh_next
            
        # 4. h_t 역전파
        # dL/dh_tilde_t
        dh_tilde_t = dh_t * cache_t['z_t']
        
        # dL/dz_t (Update Gate)
        dz_t = dh_t * (cache_t['h_tilde_t'] - cache_t['h_prev'])
        
        # 5. Candidate Hidden State (h_tilde_t) 역전파
        d_tanh_ht = dtanh(cache_t['h_tilde_t'])
        d_z_ht = dh_tilde_t * d_tanh_ht
        
        # dL/dW_x_ht, dL/db_ht
        grads['W_x_ht'] += cache_t['x_t'].T @ d_z_ht
        grads['b_ht'] += np.sum(d_z_ht, axis=0, keepdims=True)
        
        # dL/d(r_t * h_prev)
        d_r_h_prev = d_z_ht @ params['W_h_ht'].T
        
        # dL/dr_t (Reset Gate)
        dr_t = d_r_h_prev * cache_t['h_prev']
        dz_r = dr_t * dsigmoid(cache_t['r_t']) # 최종 Reset Gate 입력 그라디언트
        
        # dL/dh_prev_from_ht
        dh_prev_from_ht = d_r_h_prev * cache_t['r_t']
        
        # 6. Update Gate (z_t) 역전파
        dz_z = dz_t * dsigmoid(cache_t['z_t']) # 최종 Update Gate 입력 그라디언트
        
        # dL/dW_xz, dL/db_z
        grads['W_xz'] += cache_t['x_t'].T @ dz_z
        grads['W_hz'] += cache_t['h_prev'].T @ dz_z
        grads['b_z'] += np.sum(dz_z, axis=0, keepdims=True)
        
        # dL/dh_prev_from_z
        dh_prev_from_z = dz_z @ params['W_hz'].T
        
        # 7. Reset Gate (r_t) 역전파
        # dL/dW_xr, dL/db_r
        grads['W_xr'] += cache_t['x_t'].T @ dz_r
        grads['W_hr'] += cache_t['h_prev'].T @ dz_r
        grads['b_r'] += np.sum(dz_r, axis=0, keepdims=True)
        
        # dL/dh_prev_from_r
        dh_prev_from_r = dz_r @ params['W_hr'].T
        
        # 8. 다음 타임스텝으로 전달할 그라디언트 (dL/dh_prev)
        # dh_next = dh_t (1-z_t) + dh_prev_from_z + dh_prev_from_ht + dh_prev_from_r
        dh_next = dh_t * (1 - cache_t['z_t']) + dh_prev_from_z + dh_prev_from_ht + dh_prev_from_r
        
    return grads

def calculate_loss(Y_batch, Y_pred):
    output_size = Y_pred.shape[-1]
    N = Y_batch.shape[0]
    epsilon = 1e-8
    Y_one_hot = np.eye(output_size)[Y_batch.ravel()]
    # Cross-Entropy Loss
    loss = -np.sum(Y_one_hot * np.log(Y_pred + epsilon)) / N
    return loss

# ==========================================================
# 4. 훈련 설정 및 실행
# ==========================================================

# 하이퍼파라미터
N_SAMPLES = 1000     
SEQ_LEN = 10         
INPUT_SIZE = 5       
HIDDEN_SIZE = 32     # ********** 개선 사항: 은닉 차원 16 -> 32로 증가 **********
OUTPUT_SIZE = 2      
EPOCHS = 100         
LR = 0.001           
BATCH_SIZE = 32      

# 데이터 생성
X_data, Y_data = create_synthetic_gru_data(N_SAMPLES, SEQ_LEN, INPUT_SIZE)

# 매개변수 초기화
params = initialize_gru_params(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
# FWR-Opt 초기화
fwr_opt = FWROptimizer(lr=LR)

print(f"--- FWR-Opt를 사용한 GRU 감성 분석 모델 훈련 시작 (HIDDEN_SIZE 조정됨) ---")
print(f"샘플 수: {N_SAMPLES}, 시퀀스 길이: {SEQ_LEN}, 은닉 차원: {HIDDEN_SIZE}, LR: {LR}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

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
        try:
            Y_pred, cache = gru_forward(X_batch, params, HIDDEN_SIZE)
            loss = calculate_loss(Y_batch, Y_pred)
        except Exception as e:
            print(f"Epoch {epoch} 순전파 중 오류 발생: {e}")
            raise
            
        # 2. 역전파 실행
        try:
            grads = gru_backward(Y_batch, cache, params, OUTPUT_SIZE)
        except Exception as e:
            print(f"Epoch {epoch} 역전파 중 오류 발생: {e}")
            raise

        epoch_losses.append(loss)
        
        # 정확도 계산
        Y_pred_labels = np.argmax(Y_pred, axis=1).reshape(-1, 1)
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
try:
    Y_pred_final, _ = gru_forward(X_data, params, HIDDEN_SIZE)
    final_loss = calculate_loss(Y_data, Y_pred_final)
    
    Y_pred_labels = np.argmax(Y_pred_final, axis=1).reshape(-1, 1)
    final_accuracy = np.mean(Y_pred_labels == Y_data) * 100

    print(f"최종 평균 Cross-Entropy Loss: {final_loss:.6f}")
    print(f"최종 훈련 정확도: {final_accuracy:.2f}%")
    print("\nFWR-Opt는 GRU 모델을 사용하여 시퀀스 데이터에 대한 분류를 성공적으로 수행했습니다.")

except Exception as e:
    print(f"최종 평가 중 오류 발생: {e}")
