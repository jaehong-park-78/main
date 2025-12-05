import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. 최적화기 클래스 정의
# ==========================================================

class FWROptimizer:
    def __init__(self, lr=0.003, beta_F=0.85, beta_W=0.999, epsilon=1e-8, gamma_W=0.01, gamma_R=0.85):
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
        self.name = "FWR-Opt"
        
    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            g = grads[key]
            if key not in self.v:
                self.v[key] = np.zeros_like(g)
                self.s[key] = np.zeros_like(g)
                self.r_coherence[key] = np.ones_like(g) * 0.3

            self.v[key] = self.beta_F * self.v[key] + (1 - self.beta_F) * g
            v_hat = self.v[key] / (1 - self.beta_F**self.t)
            
            self.s[key] = self.beta_W * self.s[key] + (1 - self.beta_W) * (g**2)
            s_hat = self.s[key] / (1 - self.beta_W**self.t)
            wave_adjusted_std = np.sqrt(s_hat) + self.epsilon + self.gamma_W

            coherence_mask = np.sign(g) * np.sign(v_hat)
            
            self.r_coherence[key] = np.where(
                coherence_mask > 0,
                self.r_coherence[key] + (0.5 - self.r_coherence[key]) * 0.35,
                self.gamma_R * self.r_coherence[key]
            )
            self.r_coherence[key] = np.clip(self.r_coherence[key], 0.1, 2.5)
            
            R_factor = 1 + self.r_coherence[key]
            update_step = (self.lr * v_hat) / wave_adjusted_std * R_factor
            params[key] -= update_step
        return params

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} 
        self.v = {} 
        self.t = 0
        self.name = "Adam"
        
    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            g = grads[key]
            
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g**2)

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

# ==========================================================
# 2. 복잡한 데이터 생성 함수들
# ==========================================================

def create_complex_dataset(dataset_type='sinusoidal', n_samples=2000, noise_level=0.3):
    """다양한 복잡한 데이터셋 생성"""
    
    if dataset_type == 'sinusoidal':
        # 사인파 패턴 + 노이즈
        X = np.random.randn(n_samples, 2) * 2
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            x1, x2 = X[i]
            # 사인파 결정 경계
            if np.sin(3*x1) + 0.3*x2 + np.random.randn()*noise_level > 0:
                y[i] = 1
            else:
                y[i] = 0
                
        # 복잡성 추가: 원형 패턴 혼합
        mask = np.random.rand(n_samples) > 0.5
        circle_radius = np.sqrt(X[:,0]**2 + X[:,1]**2)
        y[mask] = (circle_radius[mask] > 1.5).astype(int)
        
    elif dataset_type == 'spiral':
        # 나선형 데이터
        n_samples_per_class = n_samples // 2
        t = np.linspace(0, 4*np.pi, n_samples_per_class)
        
        # 첫 번째 나선
        r = t * 0.5
        x1 = r * np.cos(t) + np.random.randn(n_samples_per_class) * noise_level
        y1 = r * np.sin(t) + np.random.randn(n_samples_per_class) * noise_level
        c1 = np.zeros(n_samples_per_class)
        
        # 두 번째 나선 (다른 방향)
        x2 = r * np.cos(t + np.pi) + np.random.randn(n_samples_per_class) * noise_level
        y2 = r * np.sin(t + np.pi) + np.random.randn(n_samples_per_class) * noise_level
        c2 = np.ones(n_samples_per_class)
        
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([c1, c2])
        
    elif dataset_type == 'checkerboard':
        # 체커보드 패턴
        X = np.random.rand(n_samples, 2) * 4 - 2
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            x1, x2 = X[i]
            # 체커보드 패턴
            cell_x = int(np.floor((x1 + 2) * 2)) % 2
            cell_y = int(np.floor((x2 + 2) * 2)) % 2
            y[i] = (cell_x + cell_y) % 2
            
        # 노이즈 추가
        noise_mask = np.random.rand(n_samples) < noise_level * 0.3
        y[noise_mask] = 1 - y[noise_mask]
        
    elif dataset_type == 'concentric_circles':
        # 동심원 + 노이즈
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise_level*0.5)
        X = X * 3  # 스케일 확대
        
    elif dataset_type == 'high_dim_gaussian':
        # 고차원 가우시안 믹스처
        n_features = 50
        n_classes = 4
        
        # 각 클래스별 다른 평균과 공분산
        X = []
        y = []
        
        for class_idx in range(n_classes):
            mean = np.random.randn(n_features) * 2
            cov = np.eye(n_features) * (0.5 + np.random.rand() * 2)
            
            class_samples = n_samples // n_classes
            X_class = np.random.multivariate_normal(mean, cov, class_samples)
            
            # 일부 특징은 관련이 없도록 만듦
            irrelevant_features = np.random.choice(n_features, size=n_features//4, replace=False)
            X_class[:, irrelevant_features] = np.random.randn(class_samples, len(irrelevant_features))
            
            X.append(X_class)
            y.append(np.ones(class_samples) * class_idx)
            
        X = np.vstack(X)
        y = np.hstack(y)
        
        # 이상치 추가
        n_outliers = int(n_samples * 0.05)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        X[outlier_indices] += np.random.randn(n_outliers, n_features) * 10
        
    else:
        # 기본: moons + circles 혼합
        X1, y1 = make_moons(n_samples=n_samples//2, noise=noise_level*0.3)
        X2, y2 = make_circles(n_samples=n_samples//2, noise=noise_level*0.3)
        X = np.vstack([X1 * 1.5, X2 * 2.0])
        y = np.hstack([y1, y2])
        y = (y > 0.5).astype(int)
    
    return X, y

def add_complex_noise(X, y, noise_type='mixed'):
    """다양한 노이즈 패턴 추가"""
    
    n_samples, n_features = X.shape
    
    if noise_type == 'mixed':
        # 1. 스파이크 노이즈 (극단적 아웃라이어)
        spike_samples = np.random.choice(n_samples, n_samples//20, replace=False)
        X[spike_samples] += np.random.randn(len(spike_samples), n_features) * 10
        
        # 2. 지속적 드리프트 노이즈
        drift_samples = np.random.choice(n_samples, n_samples//3, replace=False)
        drift_dir = np.random.randn(n_features)
        drift_dir = drift_dir / np.linalg.norm(drift_dir)
        for i, idx in enumerate(drift_samples):
            X[idx] += drift_dir * (i % 10) * 0.5
            
        # 3. 라벨 노이즈
        label_noise_samples = np.random.choice(n_samples, n_samples//10, replace=False)
        y[label_noise_samples] = 1 - y[label_noise_samples]
        
        # 4. 특징별 상이한 노이즈 스케일
        feature_noise_scale = np.random.rand(n_features) * 5
        for j in range(n_features):
            if np.random.rand() > 0.7:
                noise_idx = np.random.choice(n_samples, n_samples//5, replace=False)
                X[noise_idx, j] += np.random.randn(len(noise_idx)) * feature_noise_scale[j]
    
    elif noise_type == 'adversarial':
        # 적대적 노이즈: 결정 경계 근처에 집중
        from sklearn.ensemble import RandomForestClassifier
        
        # 간단한 모델로 결정 경계 추정
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        probas = clf.predict_proba(X)[:, 1]
        
        # 결정 경계 근처 샘플 선택
        boundary_threshold = 0.3
        boundary_indices = np.where((probas > 0.5 - boundary_threshold) & 
                                   (probas < 0.5 + boundary_threshold))[0]
        
        if len(boundary_indices) > 0:
            # 결정 경계를 흐리게 만드는 노이즈
            adv_noise = np.random.randn(len(boundary_indices), n_features) * 2
            # 노이즈 방향을 결정 경계의 반대 방향으로
            grad_sign = (probas[boundary_indices] > 0.5).astype(float) * 2 - 1
            adv_noise *= grad_sign[:, np.newaxis]
            X[boundary_indices] += adv_noise
    
    return X, y

# ==========================================================
# 3. 신경망 모델 및 유틸리티
# ==========================================================

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def cross_entropy_loss(Y_batch, Y_pred):
    N = Y_batch.shape[0]
    Y_pred = np.clip(Y_pred, 1e-12, 1. - 1e-12) 
    loss = -np.sum(Y_batch * np.log(Y_pred)) / N
    return loss

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def one_hot_encode(Y, num_classes):
    N = Y.shape[0]
    Y_oh = np.zeros((N, num_classes))
    Y_oh[np.arange(N), Y.astype(int)] = 1
    return Y_oh

def initialize_mlp_params(input_size, hidden_sizes, output_size):
    """동적 크기의 MLP 초기화"""
    np.random.seed(42)
    params = {}
    
    # 입력층
    params['W1'] = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size)
    params['b1'] = np.zeros((1, hidden_sizes[0]))
    
    # 은닉층들
    for i in range(1, len(hidden_sizes)):
        params[f'W{i+1}'] = np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2. / hidden_sizes[i-1])
        params[f'b{i+1}'] = np.zeros((1, hidden_sizes[i]))
    
    # 출력층
    params[f'W{len(hidden_sizes)+1}'] = np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(1. / hidden_sizes[-1])
    params[f'b{len(hidden_sizes)+1}'] = np.zeros((1, output_size))
    
    return params

def mlp_forward(X, params, output_size):
    """동적 순전파"""
    cache = {'A0': X}
    A = X
    
    # 은닉층 순전파
    num_layers = len([k for k in params.keys() if k.startswith('W')])
    
    for i in range(1, num_layers):
        W = params[f'W{i}']
        b = params[f'b{i}']
        Z = A @ W + b
        A = relu(Z)
        cache[f'Z{i}'] = Z
        cache[f'A{i}'] = A
    
    # 출력층
    W_out = params[f'W{num_layers}']
    b_out = params[f'b{num_layers}']
    Z_out = A @ W_out + b_out
    
    if output_size > 1:
        Y_pred = softmax(Z_out)
    else:
        Y_pred = sigmoid(Z_out)  # 이진 분류용
    
    cache[f'Z{num_layers}'] = Z_out
    cache['Y_pred'] = Y_pred
    
    return Y_pred, cache

def mlp_backward(Y_true, cache, params, binary=False):
    """동적 역전파"""
    grads = {}
    num_layers = len([k for k in params.keys() if k.startswith('W')])
    N = Y_true.shape[0]
    
    # 출력층 gradient
    if binary:
        # 이진 분류
        dZ = (cache['Y_pred'] - Y_true.reshape(-1, 1)) / N
    else:
        # 다중 분류
        dZ = (cache['Y_pred'] - Y_true) / N
    
    grads[f'W{num_layers}'] = cache[f'A{num_layers-1}'].T @ dZ
    grads[f'b{num_layers}'] = np.sum(dZ, axis=0, keepdims=True)
    
    dA = dZ @ params[f'W{num_layers}'].T
    
    # 은닉층 역전파
    for i in range(num_layers-1, 0, -1):
        dZ = dA * drelu(cache[f'Z{i}'])
        grads[f'W{i}'] = cache[f'A{i-1}'].T @ dZ
        grads[f'b{i}'] = np.sum(dZ, axis=0, keepdims=True)
        
        if i > 1:
            dA = dZ @ params[f'W{i}'].T
    
    return grads

# ==========================================================
# 4. 훈련 및 평가 함수
# ==========================================================

def train_model(optimizer, X_train, y_train, X_val, y_val, 
                hidden_sizes=[64, 32], epochs=100, batch_size=32, lr=0.001):
    
    input_size = X_train.shape[1]
    
    # 출력 크기 결정
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 2:
        output_size = len(unique_classes)
        binary = False
    else:
        output_size = 1
        binary = True
    
    print(f"입력 크기: {input_size}, 출력 크기: {output_size}, 이진 분류: {binary}")
    
    params = initialize_mlp_params(input_size, hidden_sizes, output_size)
    n_samples = X_train.shape[0]
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # 배치 학습
        idx = np.random.permutation(n_samples)
        epoch_loss = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = idx[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            # 순전파
            y_pred, cache = mlp_forward(X_batch, params, output_size)
            
            if binary:
                # 이진 분류
                y_batch_reshaped = y_batch.reshape(-1, 1)
                loss = binary_cross_entropy(y_batch_reshaped, y_pred)
                grads = mlp_backward(y_batch_reshaped, cache, params, binary=True)
            else:
                # 다중 분류
                y_batch_oh = one_hot_encode(y_batch, output_size)
                loss = cross_entropy_loss(y_batch_oh, y_pred)
                grads = mlp_backward(y_batch_oh, cache, params, binary=False)
            
            epoch_loss += loss
            
            # 최적화기 업데이트
            params = optimizer.update(params, grads)
        
        # 평가
        train_loss = epoch_loss / (n_samples // batch_size)
        
        # 훈련 정확도
        y_pred_train, _ = mlp_forward(X_train, params, output_size)
        if binary:
            train_acc = np.mean((y_pred_train > 0.5).flatten() == y_train) * 100
        else:
            train_acc = np.mean(np.argmax(y_pred_train, axis=1) == y_train) * 100
        
        # 검증 정확도
        y_pred_val, _ = mlp_forward(X_val, params, output_size)
        if binary:
            val_loss = binary_cross_entropy(y_val.reshape(-1, 1), y_pred_val)
            val_acc = np.mean((y_pred_val > 0.5).flatten() == y_val) * 100
        else:
            y_val_oh = one_hot_encode(y_val, output_size)
            val_loss = cross_entropy_loss(y_val_oh, y_pred_val)
            val_acc = np.mean(np.argmax(y_pred_val, axis=1) == y_val) * 100
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    return {
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_val_acc': val_accs[-1],
        'final_train_acc': train_accs[-1]
    }

# ==========================================================
# 5. 메인 실행 함수 (단일 데이터셋으로 간소화)
# ==========================================================

def run_single_experiment(dataset_type='spiral'):
    """단일 데이터셋으로 최적화기 비교 실험"""
    
    print(f"\n{'='*70}")
    print(f"데이터셋: {dataset_type.upper()}")
    print(f"{'='*70}")
    
    # 데이터 생성
    X, y = create_complex_dataset(dataset_type=dataset_type, n_samples=3000, noise_level=0.4)
    
    # 복잡한 노이즈 추가
    X, y = add_complex_noise(X, y, noise_type='mixed')
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
    print(f"클래스 분포: {np.unique(y, return_counts=True)[1]}")
    
    # 은닉층 설정
    if dataset_type == 'high_dim_gaussian':
        hidden_sizes = [128, 64, 32]
    elif dataset_type == 'spiral':
        hidden_sizes = [64, 64, 32]
    else:
        hidden_sizes = [64, 32]
    
    # 최적화기 설정
    fwr_optimizer = FWROptimizer(lr=0.003, beta_F=0.85, gamma_R=0.8)
    adam_optimizer = AdamOptimizer(lr=0.001)
    
    results = {}
    
    # FWR-Opt 훈련
    print(f"\n{'-'*50}")
    print(f"FWR-Opt 훈련 시작")
    print(f"{'-'*50}")
    
    start_time = time.time()
    fwr_result = train_model(
        fwr_optimizer, X_train, y_train, X_val, y_val,
        hidden_sizes=hidden_sizes,
        epochs=80,
        batch_size=64
    )
    fwr_time = time.time() - start_time
    fwr_result['training_time'] = fwr_time
    results['FWR-Opt'] = fwr_result
    print(f"훈련 시간: {fwr_time:.2f}초")
    
    # Adam 훈련 (동일 초기 가중치를 위해 새로운 초기화)
    print(f"\n{'-'*50}")
    print(f"Adam 훈련 시작")
    print(f"{'-'*50}")
    
    start_time = time.time()
    adam_result = train_model(
        adam_optimizer, X_train, y_train, X_val, y_val,
        hidden_sizes=hidden_sizes,
        epochs=80,
        batch_size=64
    )
    adam_time = time.time() - start_time
    adam_result['training_time'] = adam_time
    results['Adam'] = adam_result
    print(f"훈련 시간: {adam_time:.2f}초")
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 손실 곡선
    axes[0, 0].plot(fwr_result['train_losses'], 'b-', linewidth=2, label='FWR-Opt (Train)', alpha=0.7)
    axes[0, 0].plot(fwr_result['val_losses'], 'b--', linewidth=2, label='FWR-Opt (Val)', alpha=0.7)
    axes[0, 0].plot(adam_result['train_losses'], 'r-', linewidth=2, label='Adam (Train)', alpha=0.7)
    axes[0, 0].plot(adam_result['val_losses'], 'r--', linewidth=2, label='Adam (Val)', alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 정확도 곡선
    axes[0, 1].plot(fwr_result['train_accs'], 'b-', linewidth=2, label='FWR-Opt (Train)', alpha=0.7)
    axes[0, 1].plot(fwr_result['val_accs'], 'b--', linewidth=2, label='FWR-Opt (Val)', alpha=0.7)
    axes[0, 1].plot(adam_result['train_accs'], 'r-', linewidth=2, label='Adam (Train)', alpha=0.7)
    axes[0, 1].plot(adam_result['val_accs'], 'r--', linewidth=2, label='Adam (Val)', alpha=0.7)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 초기 학습 비교 (에포크 0-20)
    axes[1, 0].plot(fwr_result['train_accs'][:20], 'b-', linewidth=3, label='FWR-Opt', marker='o', markersize=4)
    axes[1, 0].plot(adam_result['train_accs'][:20], 'r-', linewidth=3, label='Adam', marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch (0-20)')
    axes[1, 0].set_ylabel('Train Accuracy (%)')
    axes[1, 0].set_title('Early Training Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 데이터 시각화 (2D 데이터만)
    if X.shape[1] == 2:
        axes[1, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          cmap='viridis', alpha=0.6, s=10, label='Train')
        axes[1, 1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, 
                          cmap='viridis', alpha=0.3, s=30, marker='x', label='Val')
        axes[1, 1].set_xlabel('Feature 1')
        axes[1, 1].set_ylabel('Feature 2')
        axes[1, 1].set_title('Dataset Visualization')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # 고차원 데이터인 경우 결과 표시
        axes[1, 1].axis('off')
        summary_text = f"Dataset: {dataset_type}\n"
        summary_text += f"Features: {X.shape[1]}\n"
        summary_text += f"Samples: {X.shape[0]}\n"
        summary_text += f"Classes: {len(np.unique(y))}\n\n"
        
        summary_text += f"FWR-Opt:\n"
        summary_text += f"  Final Val Acc: {fwr_result['final_val_acc']:.2f}%\n"
        summary_text += f"  Final Train Acc: {fwr_result['final_train_acc']:.2f}%\n"
        summary_text += f"  Training Time: {fwr_time:.2f}s\n\n"
        
        summary_text += f"Adam:\n"
        summary_text += f"  Final Val Acc: {adam_result['final_val_acc']:.2f}%\n"
        summary_text += f"  Final Train Acc: {adam_result['final_train_acc']:.2f}%\n"
        summary_text += f"  Training Time: {adam_time:.2f}s\n\n"
        
        diff_acc = fwr_result['final_val_acc'] - adam_result['final_val_acc']
        summary_text += f"Difference: {diff_acc:+.2f}% "
        if diff_acc > 0:
            summary_text += f"(FWR-Opt better by {diff_acc:.2f}%)"
        else:
            summary_text += f"(Adam better by {-diff_acc:.2f}%)"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.suptitle(f'FWR-Opt vs Adam on {dataset_type} Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"최종 결과 비교")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'FWR-Opt':<15} {'Adam':<15} {'Difference':<15}")
    print(f"{'-'*70}")
    print(f"{'Val Accuracy':<20} {fwr_result['final_val_acc']:<15.2f}% {adam_result['final_val_acc']:<15.2f}% {fwr_result['final_val_acc']-adam_result['final_val_acc']:<+15.2f}%")
    print(f"{'Train Accuracy':<20} {fwr_result['final_train_acc']:<15.2f}% {adam_result['final_train_acc']:<15.2f}% {fwr_result['final_train_acc']-adam_result['final_train_acc']:<+15.2f}%")
    print(f"{'Training Time':<20} {fwr_time:<15.2f}s {adam_time:<15.2f}s {fwr_time-adam_time:<+15.2f}s")
    
    # 최고 정확도
    fwr_best_val = max(fwr_result['val_accs'])
    adam_best_val = max(adam_result['val_accs'])
    print(f"{'Best Val Acc':<20} {fwr_best_val:<15.2f}% {adam_best_val:<15.2f}% {fwr_best_val-adam_best_val:<+15.2f}%")
    
    # 초기 학습 속도 (에포크 10 기준)
    if len(fwr_result['train_accs']) > 10 and len(adam_result['train_accs']) > 10:
        fwr_epoch10 = fwr_result['train_accs'][10]
        adam_epoch10 = adam_result['train_accs'][10]
        print(f"{'Acc @ Epoch 10':<20} {fwr_epoch10:<15.2f}% {adam_epoch10:<15.2f}% {fwr_epoch10-adam_epoch10:<+15.2f}%")
    
    print(f"{'='*70}")
    
    return results

# ==========================================================
# 6. 실행
# ==========================================================

if __name__ == "__main__":
    # 여러 데이터셋 테스트
    datasets = ['spiral', 'sinusoidal', 'checkerboard', 'concentric_circles']
    
    print("복잡한 데이터셋에서 FWR-Optimizer vs Adam 비교 실험")
    print("="*70)
    
    for dataset in datasets:
        try:
            print(f"\n{'#'*70}")
            print(f"실험 시작: {dataset}")
            print(f"{'#'*70}")
            
            results = run_single_experiment(dataset_type=dataset)
            
            # 간단한 요약
            fwr_acc = results['FWR-Opt']['final_val_acc']
            adam_acc = results['Adam']['final_val_acc']
            diff = fwr_acc - adam_acc
            
            if diff > 5:
                print(f"✓ FWR-Opt가 {diff:.2f}% 크게 우수함!")
            elif diff > 0:
                print(f"✓ FWR-Opt가 {diff:.2f}% 우수함")
            elif diff > -5:
                print(f"△ Adam이 {-diff:.2f}% 우수함")
            else:
                print(f"✗ Adam이 {-diff:.2f}% 크게 우수함")
                
            print(f"{'#'*70}\n")
            
        except Exception as e:
            print(f"데이터셋 {dataset} 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
