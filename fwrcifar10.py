### resnet18 xifar10
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler # t-SNE 전 스케일링 추가
from torchvision import datasets, transforms, models
import pickle # coarse_labels 로드를 위해 추가

# 로그 디렉토리 설정
LOG_DIR = "clustering_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- FWROptimizer 클래스 정의 (sin 함수 전용) ---
class FWROptimizerBase(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"유효하지 않은 학습률: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"유효하지 않은 beta1 값: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"유효하지 않은 beta2 값: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"유효하지 않은 beta3 값: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"유효하지 않은 weight_decay 값: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, p):
        state = self.state[p]
        state['step'] = 0
        state['flow'] = torch.zeros_like(p.data)
        state['wave'] = torch.zeros_like(p.data)
        state['resonance'] = torch.zeros_like(p.data)
        state['prev_grad'] = torch.zeros_like(p.data)

    def _update_moments(self, grad, state, beta1, beta2, beta3):
        flow = state['flow']
        wave = state['wave']
        resonance = state['resonance']
        prev_grad = state['prev_grad']

        flow.mul_(beta1).add_(grad, alpha=1 - beta1)
        wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        grad_delta = grad - prev_grad
        resonance_update = torch.sin(grad_delta)
        resonance.mul_(beta3).add_(resonance_update, alpha=1 - beta3)

    def _get_biased_corrected(self, state, step, beta1, beta2, beta3):
        flow_hat = state['flow'] / (1 - beta1**step)
        wave_hat = state['wave'] / (1 - beta2**step)
        resonance_hat = state['resonance'] / (1 - beta3**step)
        return flow_hat, wave_hat, resonance_hat

    def _apply_update(self, p, group, flow_hat, wave_hat, resonance_hat):
        denom = wave_hat.sqrt().add_(group.get('eps', 1e-8))
        update = (flow_hat + resonance_hat) / denom
        p.data.add_(-group['lr'] * update)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            eps = group.get('eps', 1e-8)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(p)

                state['step'] += 1
                step = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                self._update_moments(grad, state, beta1, beta2, beta3)
                flow_hat, wave_hat, resonance_hat = self._get_biased_corrected(state, step, beta1, beta2, beta3)
                self._apply_update(p, group, flow_hat, wave_hat, resonance_hat)
                state['prev_grad'].copy_(grad)
        return loss

class FWROptimizer(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, weight_decay)
        for group in self.param_groups:
            group.setdefault('eps', eps)

# --- 비지도 군집화를 위한 '모델' 정의 ---
class ClusterPrototypes(nn.Module):
    def __init__(self, num_clusters, data_dim):
        super(ClusterPrototypes, self).__init__()
        # 프로토타입 초기화 범위 변경 (데이터 스케일에 맞게 조절될 수 있도록)
        self.prototypes = nn.Parameter(torch.randn(num_clusters, data_dim) * 0.1)

    def forward(self, data_point):
        # 유클리드 거리 제곱 (L2 norm)
        distances_sq = torch.sum((self.prototypes - data_point)**2, dim=1)
        return distances_sq

# --- 시각화 함수 (t-SNE 사용) ---
def plot_clustering_progress(data, prototypes_history, iteration, title_suffix="", data_dim=2):
    # t-SNE는 데이터를 스케일링하는 것이 좋음
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    if data_dim > 2: # t-SNE를 적용할 경우
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, learning_rate='auto', init='random')
        data_2d = tsne.fit_transform(scaled_data)
        
        # prototypes_history는 numpy 배열이므로, 각 요소를 t-SNE 변환
        # 프로토타입도 데이터와 동일한 스케일러로 변환해야 함
        proto_2d = tsne.fit_transform(scaler.transform(prototypes_history[iteration]))
    else: # 2차원 데이터일 경우 t-SNE 없이 바로 시각화
        data_2d = scaled_data
        proto_2d = scaler.transform(prototypes_history[iteration])


    plt.figure(figsize=(10, 8)) # 그림 크기 키움
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', alpha=0.3, label='Input Data')

    # 클러스터 프로토타입 색상 고정
    colors = plt.cm.get_cmap('Dark2', len(prototypes_history[0]))
    for i, proto_coords in enumerate(proto_2d):
        plt.scatter(proto_coords[0], proto_coords[1], s=250, marker='X',
                    color=colors(i), edgecolors='black', linewidth=1.8, label=f'Prototype {i+1}')

    plt.title(f'Clustering Self-Organization ({title_suffix}) - Iteration {iteration} (t-SNE)', fontsize=14)
    plt.xlabel('Feature 1 (t-SNE)', fontsize=12)
    plt.ylabel('Feature 2 (t-SNE)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_final_clustering(data, prototypes, title_suffix="", data_dim=2, labels=None):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    if data_dim > 2: # t-SNE를 적용할 경우
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, learning_rate='auto', init='random')
        data_2d = tsne.fit_transform(scaled_data)
        proto_2d = tsne.fit_transform(scaler.transform(prototypes.cpu().numpy()))
    else: # 2차원 데이터일 경우 t-SNE 없이 바로 시각화
        data_2d = scaled_data
        proto_2d = scaler.transform(prototypes.cpu().numpy())

    plt.figure(figsize=(10, 8)) # 그림 크기 키움
    
    distances = torch.cdist(data.to(prototypes.device), prototypes)
    assignments = torch.argmin(distances, dim=1).cpu().numpy()

    # 클러스터 개수에 따른 색상 팔레트 설정
    num_clusters = prototypes.shape[0]
    # 실제 슈퍼클래스 레이블을 사용하는 경우 (0~19)
    if labels is not None:
        # 20개의 슈퍼클래스를 위한 tab20 색상 맵 사용
        unique_labels = np.unique(labels)
        for i, class_label in enumerate(unique_labels):
            points = data_2d[labels == class_label]
            if len(points) > 0:
                # tab20은 20개의 색상을 제공
                plt.scatter(points[:, 0], points[:, 1], color=plt.cm.tab20(class_label % 20), alpha=0.6, label=f'Superclass {class_label}')
    else:
        # 클러스터링 결과로 색상 표시
        colors = plt.cm.get_cmap('tab20', num_clusters) # 최대 20개 클러스터까지 다른 색상 사용
        for i in range(num_clusters):
            cluster_points = data_2d[assignments == i]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(i), alpha=0.6, label=f'Cluster {i+1}')

    for i, proto_coords in enumerate(proto_2d):
        plt.scatter(proto_coords[0], proto_coords[1], s=350, marker='X',
                    color='red', edgecolors='black', linewidth=2.5, label=f'Final Prototype {i+1}' if i==0 else "") # 첫 프로토타입만 레이블 표시

    plt.title(f'Final Self-Organized Clusters ({title_suffix}) (t-SNE)', fontsize=14)
    plt.xlabel('Feature 1 (t-SNE)', fontsize=12)
    plt.ylabel('Feature 2 (t-SNE)', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 위치 조정
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- CIFAR-100 데이터 로딩 및 특징 추출 (CNN 추가) ---
def extract_features_cifar100(cnn_model, max_samples=5000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-100 정규화
    ])
    cifar100_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # CIFAR-100 데이터 파일에서 coarse_labels 로드
    train_file = os.path.join('./data', 'cifar-100-python', 'train')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    
    # coarse_labels 추출
    coarse_labels = train_data['coarse_labels']
    
    # coarse_labels를 torch.tensor로 변환
    labels = torch.tensor(coarse_labels)

    # 데이터 샘플링 (계산 효율성)
    indices = torch.randperm(len(cifar100_dataset))[:max_samples]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar100_dataset, indices),
        batch_size=32,
        shuffle=False
    )
    labels = labels[indices] # 샘플링된 데이터에 해당하는 레이블만 사용

    # GPU 사용 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    features = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            feats = cnn_model(images)
            features.append(feats.cpu())
    
    features = torch.cat(features, dim=0)
    return features, labels

# --- CNN 클러스터링 함수 ---
def train_clustering_cifar100_with_cnn(optimizer_class, optimizer_name, num_clusters=20, num_epochs=500, feature_dim=512, use_tsne=True, tsne_dim=100, max_samples=5000, batch_size=32):
    # 1. ResNet-18 모델 로드 및 수정 (마지막 FC 레이어 제거)
    resnet = models.resnet18(pretrained=True)
    # 마지막 FC 레이어를 제거하고 512차원 특징 추출
    resnet = nn.Sequential(*list(resnet.children())[:-1]) # [N, 512, 1, 1]
    resnet.eval()

    # 2. CIFAR-100에서 특징 추출
    print("ResNet-18로 특징을 추출하는 중...")
    features, labels = extract_features_cifar100(resnet, max_samples)
    features = features.view(features.size(0), -1) # [N, 512]

    # t-SNE 적용 (선택)
    data = features
    data_dim = feature_dim
    if use_tsne:
        print(f"t-SNE를 적용하여 차원을 {tsne_dim}으로 줄이는 중...")
        # t-SNE를 적용하기 전에 데이터 스케일링
        scaler = StandardScaler()
        data_np_scaled = scaler.fit_transform(data.numpy())
        # n_components는 2 또는 3이어야 barnes_hut 알고리즘 사용 가능 (여기서는 2로 설정)
        tsne = TSNE(n_components=tsne_dim, perplexity=30, n_iter=1000, random_state=42, learning_rate='auto', init='random')
        data_np = tsne.fit_transform(data_np_scaled)
        data = torch.tensor(data_np, dtype=torch.float32)
        data_dim = tsne_dim

    # --- GPU 사용 확인 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 3. 클러스터 프로토타입 모델 초기화
    model = ClusterPrototypes(num_clusters, data_dim).to(device)

    # 4. 옵티마이저 초기화
    optimizer = optimizer_class(model.parameters(), lr=0.005, betas=(0.95, 0.999, 0.85), weight_decay=0)

    print(f" --- {optimizer_name} 클러스터링 학습 시작 (파형: sin, 차원: {data_dim}) ---")

    prototypes_history = []
    prototypes_history.append(model.prototypes.data.cpu().numpy().copy())

    # 로그 파일 설정
    log_file_path = os.path.join(LOG_DIR, f"{optimizer_name}_log_cifar100_cnn_tsne_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Clustering Log - {optimizer_name} (sin wave) on CIFAR-100 with CNN Features (t-SNE)\n")
        log_file.write(f"Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Number of Clusters: {num_clusters}\n")
        log_file.write(f"Number of Epochs: {num_epochs}\n")
        log_file.write(f"Feature Dimension: {data_dim}\n")
        log_file.write(f"Use t-SNE: {use_tsne}\n")
        log_file.write(f"Max Samples: {max_samples}\n")
        log_file.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        log_file.write(f"Betas: {optimizer.param_groups[0]['betas']}\n")
        log_file.write(f"Batch Size: {batch_size}\n\n")

    start_time = time.time()

    for epoch in tqdm(range(1, num_epochs + 1), desc=f"{optimizer_name} Training"):
        shuffled_indices = torch.randperm(data.shape[0])
        total_loss = 0.0
        num_batches = (data.shape[0] + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
            batch_data = data[shuffled_indices[start_idx:end_idx]].to(device)
            optimizer.zero_grad()

            batch_loss = 0.0
            for data_point in batch_data:
                distances_sq = model(data_point.unsqueeze(0))
                loss = torch.min(distances_sq) # 가장 가까운 프로토타입과의 거리 최소화
                batch_loss += loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / data.shape[0]

        # 정량적 평가 (너무 자주 계산하지 않도록 조정)
        if epoch % (num_epochs // 10) == 0 or epoch == 1: # 10분의 1 주기로 평가
            distances = torch.cdist(data, model.prototypes)
            assignments = torch.argmin(distances, dim=1).cpu().numpy()
            data_np = data.cpu().numpy()
            proto_np = model.prototypes.data.cpu().numpy()

            if len(np.unique(assignments)) > 1 and len(np.unique(assignments)) < len(data_np): # 실루엣 스코어 조건
                sil_score = silhouette_score(data_np, assignments)
                db_score = davies_bouldin_score(data_np, assignments)
            else:
                sil_score = db_score = "N/A (단일 클러스터 또는 너무 적은 데이터 포인트)"

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch}/{num_epochs}: Average Loss = {avg_loss:.6f}\n")
                log_file.write(f"Silhouette Score: {sil_score}\n")
                log_file.write(f"Davies-Bouldin Index: {db_score}\n")
                log_file.write("Prototypes (sample):\n")
                for idx, proto in enumerate(proto_np):
                    log_file.write(f" Prototype {idx + 1}: {proto[:5]}... ({data_dim}D)\n")
                log_file.write("\n")

            prototypes_history.append(model.prototypes.data.cpu().numpy().copy())

    end_time = time.time()
    total_time = end_time - start_time

    # 최종 정량적 평가
    distances = torch.cdist(data, model.prototypes)
    assignments = torch.argmin(distances, dim=1).cpu().numpy()
    data_np = data.cpu().numpy()
    final_sil_score = silhouette_score(data_np, assignments) if len(np.unique(assignments)) > 1 and len(np.unique(assignments)) < len(data_np) else "N/A"
    final_db_score = davies_bouldin_score(data_np, assignments) if len(np.unique(assignments)) > 1 and len(np.unique(assignments)) < len(data_np) else "N/A"

    # 최종 로그 기록
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Training Completed! Total Training Time: {total_time:.2f} seconds\n")
        log_file.write(f"Final Silhouette Score: {final_sil_score}\n")
        log_file.write(f"Final Davies-Bouldin Index: {final_db_score}\n")
        log_file.write("Final Prototypes:\n")
        for idx, proto in enumerate(model.prototypes.data.cpu().numpy()):
            log_file.write(f" Prototype {idx + 1}: {proto[:5]}... ({data_dim}D)\n")

    print(f" 클러스터링 학습 완료! 총 학습 시간: {total_time:.2f} 초")
    print(f"Final Silhouette Score: {final_sil_score}")
    print(f"Final Davies-Bouldin Index: {final_db_score}")
    print(f"로그 파일이 저장되었습니다: {log_file_path}")

    final_prototypes = model.prototypes.data.cpu().numpy()
    return data.cpu(), final_prototypes, prototypes_history, f"{optimizer_name} (sin wave)", total_time, labels.cpu()

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # 시드 고정 (재현성을 위해)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    num_clusters = 20 # CIFAR-100 슈퍼클래스 기준으로 20개 클러스터
    num_epochs = 500
    feature_dim = 512 # ResNet-18의 출력 차원
    use_tsne = True # t-SNE로 차원 축소
    tsne_dim = 2 # t-SNE로 축소할 차원 수 (barnes_hut 알고리즘의 제약사항 때문에 2 또는 3으로 설정)
    max_samples = 5000 # 학습 데이터 샘플 수

    # CIFAR-100 데이터로 학습 (CNN 특징 사용, t-SNE 적용)
    data_points_cpu, final_protos_sin, history_sin, title_sin, time_sin, labels = train_clustering_cifar100_with_cnn(
        FWROptimizer, "FWROptimizer", num_clusters=num_clusters, num_epochs=num_epochs,
        feature_dim=feature_dim, use_tsne=use_tsne, tsne_dim=tsne_dim, max_samples=max_samples
    )

    # 학습 과정 시각화 (중간 단계와 마지막)
    print("\n클러스터링 진행 상황 시각화 (초기)...")
    plot_clustering_progress(data_points_cpu.numpy(), history_sin, 0, title_sin, tsne_dim if use_tsne else feature_dim)
    print("\n클러스터링 진행 상황 시각화 (최종)...")
    plot_clustering_progress(data_points_cpu.numpy(), history_sin, -1, title_sin, tsne_dim if use_tsne else feature_dim)

    # 최종 클러스터링 결과 시각화 (실제 슈퍼클래스 레이블로 색상 표시)
    print("\n최종 클러스터링 결과 시각화 (실제 슈퍼클래스 레이블)...")
    plot_final_clustering(data_points_cpu, torch.tensor(final_protos_sin), title_sin + " with Superclass Labels",
                         tsne_dim if use_tsne else feature_dim, labels=labels.numpy())

    # 클러스터링 결과 시각화 (클러스터링 레이블로 색상 표시)
    print("\n최종 클러스터링 결과 시각화 (클러스터링 레이블)...")
    plot_final_clustering(data_points_cpu, torch.tensor(final_protos_sin), title_sin + " with Clustered Labels",
                         tsne_dim if use_tsne else feature_dim)

    print(f"\nFWROptimizer (sin) 최종 프로토타입 (일부): {final_protos_sin[0, :5]}...")
