import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
# Import from torch.amp directly, as suggested by the warnings
from torch.amp import autocast, GradScaler
from sklearn.metrics import silhouette_score
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights # Use recommended weights for ResNet

# --- FWROptimizerBase 클래스 정의 ---
class _FWROptimizerBase(torch.optim.Optimizer):
    """
    FWROptimizer의 기본 클래스입니다.
    이 클래스는 FWR (Flow, Wave, Resonance) 최적화 알고리즘의 핵심 로직을 정의합니다.
    """
    def __init__(
        self,
        params: list,
        lr: float = 0.001,
        betas: tuple = (0.95, 0.999, 0.9),
        weight_decay: float = 0,
        waveform: str = "sin",
        max_grad_norm: float = 0.5
    ):
        """
        FWROptimizerBase를 초기화합니다.

        Args:
            params (list): 최적화할 모델 파라미터.
            lr (float): 학습률. (기본값: 0.001)
            betas (tuple): FWR 모멘트 계산을 위한 계수 (beta1, beta2, beta3). (기본값: (0.95, 0.999, 0.9))
            weight_decay (float): 가중치 감쇠 (L2 페널티). (기본값: 0)
            waveform (str): 공명 항을 계산하는 데 사용되는 파형 ("sin", "tanh", "cos", "sawtooth"). (기본값: "sin")
            max_grad_norm (float): 그래디언트 클리핑을 위한 최대 L2 노름 값. (기본값: 0.5)

        Raises:
            ValueError: 유효하지 않은 입력 파라미터가 감지될 경우.
        """
        if not 0.0 <= lr:
            raise ValueError(f"유효하지 않은 학습률: {lr}")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"유효하지 않은 beta 값: {betas}. 모든 beta는 [0.0, 1.0) 범위에 있어야 합니다.")
        if not 0.0 <= weight_decay:
            raise ValueError(f"유효하지 않은 weight_decay 값: {weight_decay}")
        if waveform not in ["sin", "tanh", "cos", "sawtooth"]:
            raise ValueError(f"유효하지 않은 waveform: {waveform}. 'sin', 'tanh', 'cos', 'sawtooth' 중 선택하세요.")
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"유효하지 않은 max_grad_norm 값: {max_grad_norm}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            waveform=waveform,
            max_grad_norm=max_grad_norm
        )
        super().__init__(params, defaults)

    def _init_state(self, p: torch.Tensor):
        """
        주어진 파라미터 p에 대한 옵티마이저 상태를 초기화합니다.
        """
        state = self.state[p]
        state['step'] = 0
        state['flow'] = torch.zeros_like(p.data)
        state['wave'] = torch.zeros_like(p.data)
        state['resonance'] = torch.zeros_like(p.data)
        state['prev_grad'] = torch.zeros_like(p.data)

    def _update_moments(
        self,
        grad: torch.Tensor,
        state: dict,
        beta1: float,
        beta2: float,
        beta3: float,
        waveform: str
    ):
        """
        FWR 모멘트(flow, wave, resonance)를 업데이트합니다.
        """
        flow = state['flow']
        wave = state['wave']
        resonance = state['resonance']
        prev_grad = state['prev_grad']

        flow.mul_(beta1).add_(grad, alpha=1 - beta1)
        wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        grad_delta = grad - prev_grad

        if waveform == "sin":
            resonance_update = torch.sin(grad_delta)
        elif waveform == "tanh":
            resonance_update = torch.tanh(grad_delta)
        elif waveform == "cos":
            resonance_update = torch.cos(grad_delta) * 0.5  # 스케일링
        elif waveform == "sawtooth":
            # 톱니파 함수: (x / pi) - floor(x / (2 * pi) + 0.5) * 2
            # torch.remainder를 사용하여 0에서 2*pi 사이로 정규화
            resonance_update = (grad_delta / torch.pi) - 2 * torch.floor(grad_delta / (2 * torch.pi) + 0.5)

        resonance.mul_(beta3).add_(resonance_update, alpha=1 - beta3)

    def _get_biased_corrected(
        self,
        state: dict,
        step: int,
        beta1: float,
        beta2: float,
        beta3: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        바이어스 보정된 FWR 모멘트를 계산합니다.
        """
        flow_hat = state['flow'] / (1 - beta1**step)
        wave_hat = state['wave'] / (1 - beta2**step)
        resonance_hat = state['resonance'] / (1 - beta3**step)
        return flow_hat, wave_hat, resonance_hat

    def _apply_update(
        self,
        p: torch.Tensor,
        group: dict,
        flow_hat: torch.Tensor,
        wave_hat: torch.Tensor,
        resonance_hat: torch.Tensor
    ):
        """
        파라미터에 업데이트를 적용합니다.
        """
        denom = wave_hat.sqrt().add_(group['eps']) # eps는 이미 group에 설정되어 있음
        update = (flow_hat + resonance_hat) / denom
        p.data.add_(-group['lr'] * update)

    @torch.no_grad()
    def step(self, closure=None):
        """
        최적화 단계를 수행합니다.

        Args:
            closure (callable, optional): 모델을 재평가하고 손실을 반환하는 클로저.

        Returns:
            torch.Tensor: 클로저가 제공된 경우의 손실.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            waveform = group['waveform']
            max_grad_norm = group['max_grad_norm']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_([p], max_norm=max_grad_norm)

                state = self.state[p]

                if len(state) == 0:
                    self._init_state(p)

                state['step'] += 1
                step = state['step']

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                self._update_moments(grad, state, beta1, beta2, beta3, waveform)
                flow_hat, wave_hat, resonance_hat = self._get_biased_corrected(state, step, beta1, beta2, beta3)
                self._apply_update(p, group, flow_hat, wave_hat, resonance_hat)

                # 이전 그래디언트 저장 (다음 스텝의 grad_delta 계산용)
                state['prev_grad'].copy_(grad)
        return loss

class FWROptimizer(_FWROptimizerBase):
    """
    Flow, Wave, Resonance (FWR) 최적화 알고리즘입니다.
    """
    def __init__(
        self,
        params: list,
        lr: float = 0.001,
        betas: tuple = (0.95, 0.999, 0.9),
        eps: float = 1e-8,
        weight_decay: float = 0,
        waveform: str = "sin",
        max_grad_norm: float = 0.5
    ):
        """
        FWROptimizer를 초기화합니다.

        Args:
            params (list): 최적화할 모델 파라미터.
            lr (float): 학습률. (기본값: 0.001)
            betas (tuple): FWR 모멘트 계산을 위한 계수 (beta1, beta2, beta3). (기본값: (0.95, 0.999, 0.9))
            eps (float): 0으로 나누는 것을 방지하기 위한 작은 값. (기본값: 1e-8)
            weight_decay (float): 가중치 감쇠 (L2 페널티). (기본값: 0)
            waveform (str): 공명 항을 계산하는 데 사용되는 파형 ("sin", "tanh", "cos", "sawtooth"). (기본값: "sin")
            max_grad_norm (float): 그래디언트 클리핑을 위한 최대 L2 노름 값. (기본값: 0.5)
        """
        super().__init__(params, lr, betas, weight_decay, waveform, max_grad_norm)
        # 모든 parameter group에 'eps' 기본값 설정
        for group in self.param_groups:
            group.setdefault('eps', eps)

# --- 비지도 군집화를 위한 '모델' 정의 ---
class ClusterPrototypes(nn.Module):
    """
    비지도 군집화를 위한 클러스터 프로토타입을 정의하는 모델입니다.
    각 데이터 포인트와 클러스터 프로토타입 간의 유클리드 거리를 계산합니다.
    """
    def __init__(self, num_clusters: int, data_dim: int):
        """
        ClusterPrototypes 모델을 초기화합니다.

        Args:
            num_clusters (int): 클러스터의 수.
            data_dim (int): 입력 데이터의 차원.
        """
        super(ClusterPrototypes, self).__init__()
        # 클러스터 프로토타입을 -1과 1 사이의 값으로 초기화합니다.
        self.prototypes = nn.Parameter(torch.rand(num_clusters, data_dim) * 2 - 1)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터 배치와 프로토타입 간의 제곱 유클리드 거리를 계산합니다.

        Args:
            data_batch (torch.Tensor): 입력 데이터 배치.

        Returns:
            torch.Tensor: 각 데이터 포인트와 모든 프로토타입 간의 제곱 거리.
        """
        # torch.cdist는 (B, P, M)과 (B, R, M) 텐서를 받아 (B, P, R) 텐서를 반환합니다.
        # 따라서 unsqueeze(0)를 사용하여 배치 차원을 추가합니다.
        distances_sq = torch.cdist(data_batch.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0) ** 2
        return distances_sq

# --- 학습 및 평가 함수 ---
def train_clustering(
    optimizer_class: type[torch.optim.Optimizer],
    optimizer_name: str,
    num_clusters: int = 10,
    num_epochs: int = 100,
    lr: float = 0.01,
    waveform: str = "sin",
    batch_size: int = 256,
    num_samples: int = 10000 # 추출할 샘플 수 파라미터 추가
) -> tuple[float, float, float, str]:
    """
    주어진 옵티마이저를 사용하여 비지도 군집화를 학습하고 평가합니다.

    Args:
        optimizer_class (type[torch.optim.Optimizer]): 사용할 옵티마이저 클래스 (예: FWROptimizer, torch.optim.Adam).
        optimizer_name (str): 옵티마이저의 이름 (출력용).
        num_clusters (int): 클러스터의 수. (기본값: 10)
        num_epochs (int): 학습 에포크 수. (기본값: 100)
        lr (float): 학습률. (기본값: 0.01)
        waveform (str): FWROptimizer에 사용될 파형. (기본값: "sin")
        batch_size (int): 학습 배치 크기. (기본값: 256)
        num_samples (int): CIFAR-10에서 특성 추출에 사용할 샘플 수. (기본값: 10000)

    Returns:
        tuple[float, float, float, str]: 총 학습 시간, 최종 평균 손실, 실루엣 점수, 옵티마이저 이름.
    """
    # 1. CIFAR-10 데이터셋 로드 및 특성 추출
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]로 정규화
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 지정된 수의 샘플을 무작위로 선택
    if num_samples > len(dataset):
        print(f"경고: 요청된 샘플 수 ({num_samples})가 데이터셋 크기 ({len(dataset)})보다 큽니다. 전체 데이터셋을 사용합니다.")
        num_samples = len(dataset)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    # 데이터 로더를 사용하여 특성 추출 효율화
    feature_extractor_dataloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f" --- GPU 사용 가능! CUDA 장치: {torch.cuda.get_device_name(0)} ---")
        scaler = GradScaler()
    else:
        print(" --- 경고: GPU (CUDA)를 사용할 수 없습니다! CPU로 실행됩니다. ---")
        scaler = None

    # ResNet-18로 특성 추출 (사전 학습된 가중치 사용)
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    resnet.fc = nn.Identity() # 분류 계층 제거

    all_features = []
    print(f" --- ResNet-18로 {num_samples}개 샘플에서 특성 추출 중... ---")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(feature_extractor_dataloader, desc="Feature Extraction")):
            images = images.to(device)
            features = resnet(images).view(images.size(0), -1)
            all_features.append(features.cpu())
    data = torch.cat(all_features, dim=0) # (num_samples, 512)
    data_dim = data.shape[1]

    # 2. 클러스터 프로토타입 모델 초기화
    torch.manual_seed(42) # 재현성을 위해 시드 설정
    model = ClusterPrototypes(num_clusters, data_dim).to(device)

    # 3. 옵티마이저 초기화
    if optimizer_class == FWROptimizer:
        # 특정 파형에 따라 LR 다르게 설정
        current_lr = lr if waveform in ["sin", "tanh"] else 0.0001
        optimizer = optimizer_class(
            model.parameters(),
            lr=current_lr,
            betas=(0.95, 0.999, 0.9),
            weight_decay=0,
            waveform=waveform,
            max_grad_norm=0.5
        )
        print(f"   FWROptimizer ({waveform}) 초기화됨. 학습률: {current_lr}")
    elif optimizer_class == torch.optim.Adam:
        optimizer = optimizer_class(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0
        )
        print(f"   Adam 옵티마이저 초기화됨. 학습률: {lr}")
    elif optimizer_class == torch.optim.SGD:
        optimizer = optimizer_class(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0
        )
        print(f"   SGD 옵티마이저 초기화됨. 학습률: {lr}")
    else:
        raise ValueError(f"지원하지 않는 옵티마이저 클래스: {optimizer_class}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    print(f" --- {optimizer_name} 클러스터링 학습 시작 ---")

    start_time = time.time()

    # 학습 데이터 로더 (특성 추출된 데이터를 사용)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(1, num_epochs + 1), desc=f"{optimizer_name} Training")
    for epoch in pbar:
        total_loss = 0.0
        num_processed_samples = 0

        for data_batch in train_dataloader:
            data_batch = data_batch.to(device) # 데이터를 장치로 이동
            optimizer.zero_grad()

            # --- CORRECTED LINE FOR autocast ---
            # Use torch.amp.autocast('cuda', dtype=torch.float16) as per warnings
            # The 'enabled' argument is automatically handled if the first arg is the device type
            with autocast(device.type, dtype=torch.float16):
                distances_sq = model(data_batch)
                # 각 데이터 포인트에 가장 가까운 프로토타입까지의 거리 제곱을 손실로 사용
                loss = torch.min(distances_sq, dim=1)[0].mean()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data_batch.size(0)
            num_processed_samples += data_batch.size(0)

        avg_loss = total_loss / num_processed_samples
        scheduler.step(avg_loss)
        pbar.set_postfix(avg_loss=f'{avg_loss:.6f}', lr=optimizer.param_groups[0]['lr'])

    end_time = time.time()
    total_time = end_time - start_time
    print(f" 클러스터링 학습 완료! 총 학습 시간: {total_time:.2f} 초")

    # 최종 클러스터 할당 및 실루엣 점수 계산
    model.eval() # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
    with torch.no_grad():
        final_prototypes = model.prototypes.data
        # 전체 데이터에 대한 할당을 위해 데이터를 다시 장치로 이동
        data_on_device = data.to(device)
        distances = torch.cdist(data_on_device, final_prototypes)
        assignments = torch.argmin(distances, dim=1).cpu().numpy()

    # 실루엣 점수 계산 (단일 클러스터가 아닐 경우에만)
    unique_clusters = np.unique(assignments)
    if len(unique_clusters) > 1:
        sil_score = silhouette_score(data.cpu().numpy(), assignments)
    else:
        sil_score = -1.0 # 모든 샘플이 단일 클러스터에 할당된 경우 실루엣 점수 계산 불가
        print("경고: 모든 샘플이 단일 클러스터에 할당되어 실루엣 점수를 계산할 수 없습니다. -1.0으로 설정됩니다.")

    # 클러스터 할당 비율
    cluster_counts = np.bincount(assignments, minlength=num_clusters)
    print(f"Cluster Assignment Counts: {cluster_counts}")

    return total_time, avg_loss, sil_score, optimizer_name

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # 하이퍼파라미터 설정
    NUM_CLUSTERS = 10
    NUM_EPOCHS = 100 # 충분한 반복을 위해 100으로 설정
    LEARNING_RATE = 0.01
    BATCH_SIZE = 256
    NUM_SAMPLES_FOR_FEATURES = 10000 # 특성 추출에 사용할 샘플 수 (메모리 제약 고려)

    optimizers_results = []

    print(f"\n===== 클러스터링 실험 시작 (총 {NUM_SAMPLES_FOR_FEATURES}개 샘플 사용) =====")

    # 1. FWROptimizer with different waveforms
    waveforms = ["sin", "tanh", "cos", "sawtooth"]
    for waveform_type in waveforms:
        print(f"\n--- FWROptimizer ({waveform_type}) 실험 ---")
        time_res, loss_res, sil_score_res, name_res = train_clustering(
            FWROptimizer,
            f"FWROptimizer ({waveform_type})",
            num_clusters=NUM_CLUSTERS,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            waveform=waveform_type,
            batch_size=BATCH_SIZE,
            num_samples=NUM_SAMPLES_FOR_FEATURES
        )
        optimizers_results.append((name_res, time_res, loss_res, sil_score_res))

    # 2. Adam
    print("\n--- Adam 옵티마이저 실험 ---")
    time_res, loss_res, sil_score_res, name_res = train_clustering(
        torch.optim.Adam,
        "Adam",
        num_clusters=NUM_CLUSTERS,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_FOR_FEATURES
    )
    optimizers_results.append((name_res, time_res, loss_res, sil_score_res))

    # 3. SGD
    print("\n--- SGD 옵티마이저 실험 ---")
    time_res, loss_res, sil_score_res, name_res = train_clustering(
        torch.optim.SGD,
        "SGD",
        num_clusters=NUM_CLUSTERS,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES_FOR_FEATURES
    )
    optimizers_results.append((name_res, time_res, loss_res, sil_score_res))

    # 학습 결과 출력
    print("\n" + "="*40)
    print("--- Optimizer Comparison Results ---")
    print("="*40)
    for name, time_val, loss_val, sil_score_val in optimizers_results:
        print(f"\nOptimizer: {name}")
        print(f"  Training Time: {time_val:.2f} seconds")
        print(f"  Final Average Loss: {loss_val:.6f}")
        print(f"  Silhouette Score: {sil_score_val:.4f}")
    print("\n===== 실험 종료 =====")
