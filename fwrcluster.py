#자기 조직화
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm  # tqdm 라이브러리 임포트!

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

# --- 비지도 군집화를 위한 '모델' 정의 (클러스터 중심) ---
class ClusterPrototypes(nn.Module):
    def __init__(self, num_clusters, data_dim):
        super(ClusterPrototypes, self).__init__()
        self.prototypes = nn.Parameter(torch.rand(num_clusters, data_dim) * 2 - 1)

    def forward(self, data_point):
        distances_sq = torch.sum((self.prototypes - data_point)**2, dim=1)
        return distances_sq

# --- 시각화 함수 ---
def plot_clustering_progress(data, prototypes_history, iteration, title_suffix=""):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.3, label='Input Data')

    colors = plt.cm.get_cmap('Dark2', len(prototypes_history[0]))
    for i, proto_coords in enumerate(prototypes_history[iteration]):
        plt.scatter(proto_coords[0], proto_coords[1], s=200, marker='X',
                    color=colors(i), edgecolors='black', linewidth=1.5,
                    label=f'Prototype {i+1}')

    plt.title(f'Clustering Self-Organization ({title_suffix}) - Iteration {iteration}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def plot_final_clustering(data, prototypes, title_suffix=""):
    plt.figure(figsize=(8, 8))

    distances = torch.cdist(data.to(prototypes.device), prototypes)
    assignments = torch.argmin(distances, dim=1)

    colors = plt.cm.get_cmap('Dark2', prototypes.shape[0])
    for i in range(prototypes.shape[0]):
        cluster_points = data[assignments == i].cpu().numpy()
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(i), alpha=0.6, label=f'Cluster {i+1}')

    for i, proto_coords in enumerate(prototypes.cpu().numpy()):
        plt.scatter(proto_coords[0], proto_coords[1], s=300, marker='X',
                    color=colors(i), edgecolors='black', linewidth=2, label=f'Final Prototype {i+1}')

    plt.title(f'Final Self-Organized Clusters ({title_suffix})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

# --- 학습 및 평가 함수 (수정) ---
def train_clustering(optimizer_class, optimizer_name, num_clusters=3, num_epochs=500):  # 기본 Epoch 500으로 설정
    # 1. 랜덤 데이터 생성 (클러스터가 있는 2D 데이터)
    num_samples = 1000
    np.random.seed(42)

    data1 = np.random.randn(num_samples // 3, 2) * 0.2 + np.array([0.7, 0.7])
    data2 = np.random.randn(num_samples // 3, 2) * 0.2 + np.array([-0.7, 0.7])
    data3 = np.random.randn(num_samples // 3, 2) * 0.2 + np.array([0.0, -0.7])

    data_np = np.vstack((data1, data2, data3))
    np.random.shuffle(data_np)
    data = torch.tensor(data_np, dtype=torch.float32)

    data_dim = data.shape[1]

    # --- GPU 사용 강제 및 확인 ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" --- GPU 사용 가능! CUDA 장치: {torch.cuda.get_device_name(0)} ---")
    else:
        device = torch.device('cpu')
        print(" --- 경고: GPU (CUDA)를 사용할 수 없습니다! CPU로 실행됩니다. ---")
        print(" GPU를 사용하려면 CUDA 드라이버 및 PyTorch 설치를 확인하세요.")
    # GPU 사용이 필수라면 여기서 프로그램 종료 가능:
    # raise RuntimeError("CUDA를 사용할 수 없어 GPU 학습을 진행할 수 없습니다.")
    # --- GPU 사용 강제 및 확인 끝 ---

    data = data.to(device)  # 데이터도 GPU로 이동

    # 2. 클러스터 프로토타입 모델 초기화
    model = ClusterPrototypes(num_clusters, data_dim).to(device)  # 모델도 GPU로 이동

    # 3. 옵티마이저 초기화 (FWROptimizer)
    optimizer = optimizer_class(
        model.parameters(),
        lr=0.01,
        betas=(0.95, 0.999, 0.9),
        weight_decay=0
    )

    print(f" --- {optimizer_name} 클러스터링 학습 시작 (파형: sin) ---")

    prototypes_history = []
    prototypes_history.append(model.prototypes.data.cpu().numpy().copy())

    start_time = time.time()

    # tqdm 적용! num_epochs를 progress bar로 감싸서 진행률을 표시
    # Create the tqdm object and store it in a variable (e.g., pbar)
    pbar = tqdm(range(1, num_epochs + 1), desc=f"{optimizer_name} Training")
    for epoch in pbar: # Iterate using the pbar object
        shuffled_indices = torch.randperm(data.shape[0])
        total_loss = 0.0

        for i in shuffled_indices:
            data_point = data[i]
            optimizer.zero_grad()

            distances_sq = model(data_point)
            loss = torch.min(distances_sq)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / data.shape[0]

        # tqdm 바에 현재 손실 업데이트
        # Call set_postfix on the pbar object
        pbar.set_postfix(avg_loss=f'{avg_loss:.6f}')

        # 스냅샷 저장 빈도 조정
        if epoch % (num_epochs // 5) == 0 or epoch == 1:
            prototypes_history.append(model.prototypes.data.cpu().numpy().copy())
        # tqdm 때문에 이 부분의 print는 주석 처리하거나 필요시 사용
        # print(f"Epoch {epoch:4d}/{num_epochs}: Average Loss = {avg_loss:.6f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f" 클러스터링 학습 완료! 총 학습 시간: {total_time:.2f} 초")

    final_prototypes = model.prototypes.data.cpu().numpy()

    return data.cpu(), final_prototypes, prototypes_history, f"{optimizer_name} (sin wave)", total_time

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    num_clusters = 3
    num_iterations = 500  # 더 빠른 결과를 위해 Epoch 유지

    # FWROptimizer (sin 파형)으로만 학습
    data_points_cpu, final_protos_sin, history_sin, title_sin, time_sin = train_clustering(
        FWROptimizer, "FWROptimizer",
        num_clusters=num_clusters, num_epochs=num_iterations
    )

    # 학습 과정 시각화 (중간 단계와 마지막)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, 0, title_sin)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, 1, title_sin)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, 2, title_sin)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, 3, title_sin)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, 4, title_sin)
    plot_clustering_progress(data_points_cpu.cpu().numpy(), history_sin, -1, title_sin)  # 최종

    # 최종 클러스터링 결과 시각화
    plot_final_clustering(data_points_cpu, torch.tensor(final_protos_sin), title_sin)

    print(f" FWROptimizer (sin) 최종 프로토타입: {final_protos_sin}")
