# SVHN _ 심화
#!pip install scikit-image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- Sawtooth 파형 함수 정의 (참고용, sin만 사용) ---
# 기존 코드와 동일하게 유지. sin, cos, tanh와 함께 사용할 수 있도록 sawtooth 함수가 필요해.
def sawtooth(x, period=1.0):
    """
    Sawtooth 파형을 생성합니다.
    Args:
        x (torch.Tensor): 입력 텐서.
        period (float): 파형의 주기.
    Returns:
        torch.Tensor: Sawtooth 파형 값.
    """
    return 2 * (x / period - torch.floor(0.5 + x / period))

# --- FWR 옵티마이저 기본 클래스 ---
class FWROptimizerBase(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=0, wave_function='sin'):
        """
        FWROptimizer의 기본 클래스를 초기화합니다.
        Args:
            params (iterable): 최적화할 파라미터.
            lr (float): 학습률.
            betas (tuple): (beta1, beta2, beta3) 모멘트 계수.
            weight_decay (float): 가중치 감쇠 (L2 페널티).
            wave_function (str): 'sin', 'cos', 'tanh', 'sawtooth' 중 사용할 파형 함수.
        """
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

        valid_functions = ['sin', 'cos', 'tanh', 'sawtooth']
        if wave_function not in valid_functions:
            raise ValueError(f"wave_function은 {valid_functions} 중 하나여야 합니다, 입력: {wave_function}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, wave_function=wave_function)
        super().__init__(params, defaults)

    def _init_state(self, p):
        """
        주어진 파라미터 `p`에 대한 옵티마이저 상태를 초기화합니다.
        """
        state = self.state[p]
        state['step'] = 0
        state['flow'] = torch.zeros_like(p.data)  # 첫 번째 모멘트 (평균 기울기)
        state['wave'] = torch.zeros_like(p.data)  # 두 번째 모멘트 (제곱 기울기)
        state['resonance'] = torch.zeros_like(p.data)  # 공명 항
        state['prev_grad'] = torch.zeros_like(p.data)  # 이전 기울기 (resonance 계산용)

    def _update_moments(self, grad, state, beta1, beta2, beta3, wave_function):
        """
        주어진 기울기와 현재 상태를 기반으로 모멘트(flow, wave, resonance)를 업데이트합니다.
        """
        flow = state['flow']
        wave = state['wave']
        resonance = state['resonance']
        prev_grad = state['prev_grad']

        # 첫 번째 모멘트 (Adam의 m과 유사)
        flow.mul_(beta1).add_(grad, alpha=1 - beta1)

        # 두 번째 모멘트 (Adam의 v와 유사)
        wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # 기울기 변화량
        grad_delta = grad - prev_grad

        # 공명 항 업데이트 (선택된 파형 함수 사용)
        if wave_function == 'sin':
            resonance_update = torch.sin(grad_delta)
        elif wave_function == 'cos':
            resonance_update = torch.cos(grad_delta)
        elif wave_function == 'tanh':
            resonance_update = torch.tanh(grad_delta)
        elif wave_function == 'sawtooth':
            resonance_update = sawtooth(grad_delta)  # sawtooth 함수 호출
        else:  # 이미 __init__에서 검증했지만, 만약을 위해.
            raise ValueError(f"알 수 없는 파형 함수: {wave_function}")

        resonance.mul_(beta3).add_(resonance_update, alpha=1 - beta3)

    def _get_biased_corrected(self, state, step, beta1, beta2, beta3):
        """
        바이어스 보정된 모멘트 추정값을 계산합니다.
        Adam 옵티마이저와 유사한 바이어스 보정 방식입니다.
        """
        flow_hat = state['flow'] / (1 - beta1**step)
        wave_hat = state['wave'] / (1 - beta2**step)
        resonance_hat = state['resonance'] / (1 - beta3**step)
        return flow_hat, wave_hat, resonance_hat

    def _apply_update(self, p, group, flow_hat, wave_hat, resonance_hat):
        """
        바이어스 보정된 모멘트를 사용하여 파라미터 `p`를 업데이트합니다.
        """
        # 분모에 작은 값(eps)을 더해 0으로 나누는 것을 방지
        denom = wave_hat.sqrt().add_(group.get('eps', 1e-8))
        update = (flow_hat + resonance_hat) / denom
        p.data.add_(-group['lr'] * update)  # lr * update로 수정

    @torch.no_grad()
    def step(self, closure=None):
        """
        옵티마이저의 한 스텝을 수행합니다.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            wave_function = group['wave_function']
            # 각 파라미터 그룹에 대한 epsilon 값 가져오기
            eps = group.get('eps', 1e-8)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 상태가 초기화되지 않았다면 초기화
                if len(state) == 0:
                    self._init_state(p)

                state['step'] += 1
                step = state['step']

                # 가중치 감쇠 적용 (L2 페널티)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # 모멘트 업데이트
                self._update_moments(grad, state, beta1, beta2, beta3, wave_function)

                # 바이어스 보정된 모멘트 계산
                flow_hat, wave_hat, resonance_hat = self._get_biased_corrected(state, step, beta1, beta2, beta3)

                # 파라미터 업데이트
                self._apply_update(p, group, flow_hat, wave_hat, resonance_hat)

                # 다음 스텝을 위해 현재 기울기 저장
                state['prev_grad'].copy_(grad)
        return loss

# --- FWROptimizer 클래스 (FWROptimizerBase를 상속받아 사용) ---
class FWROptimizer(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, weight_decay=0, wave_function='sin'):
        """
        FWROptimizer를 초기화합니다.
        FWROptimizerBase에 epsilon(eps) 매개변수를 추가합니다.
        Args:
            params (iterable): 최적화할 파라미터.
            lr (float): 학습률.
            betas (tuple): (beta1, beta2, beta3) 모멘트 계수.
            eps (float): 0으로 나누는 것을 방지하기 위한 작은 값.
            weight_decay (float): 가중치 감쇠 (L2 페널티).
            wave_function (str): 'sin', 'cos', 'tanh', 'sawtooth' 중 사용할 파형 함수.
        """
        super().__init__(params, lr, betas, weight_decay, wave_function)
        # 각 파라미터 그룹에 eps 기본값 설정
        for group in self.param_groups:
            group.setdefault('eps', eps)

# --- 컨볼루션 오토인코더 모델 정의 (SVHN용) ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        """
        SVHN 데이터셋 (컬러 이미지 3채널)을 처리하기 위한 컨볼루션 오토인코더 모델을 정의합니다.
        """
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 입력: [3, 32, 32] -> 출력: [16, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 출력: [16, 16, 16]
            nn.Conv2d(16, 8, 3, padding=1),  # 출력: [8, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 출력: [8, 8, 8] (잠재 공간)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # 입력: [8, 8, 8] -> 출력: [16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 출력: [8, 32, 32]
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  # 출력: [3, 32, 32]
            nn.Sigmoid()  # 이미지 픽셀 값을 [0, 1] 범위로 출력하기 위해 Sigmoid 사용
        )

    def forward(self, x):
        """
        오토인코더의 포워드 패스를 정의합니다.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 데이터 로딩 (SVHN) ---
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환하고 픽셀 값을 [0, 1] 범위로 정규화
])

# SVHN 데이터셋 다운로드 및 로드
train_dataset = SVHN(root='./data', split='train', transform=transform, download=True)
test_dataset = SVHN(root='./data', split='test', transform=transform, download=True)

# 데이터 로더 설정
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# --- SSIM 계산 함수 (컬러 이미지용) ---
def calculate_ssim(model, test_loader, device):
    """
    모델의 재구성된 이미지와 원본 이미지 간의 SSIM (Structural Similarity Index Measure)을 계산합니다.
    컬러 이미지의 경우 RGB 채널별 SSIM을 계산한 후 평균을 냅니다.
    """
    model.eval()  # 모델을 평가 모드로 설정
    ssim_scores = []
    with torch.no_grad():  # 기울기 계산 비활성화
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i in range(inputs.size(0)):
                # 텐서를 NumPy 배열로 변환하고 채널 순서 변경: [C, H, W] -> [H, W, C]
                img_orig = inputs[i].cpu().numpy().transpose(1, 2, 0)
                img_recon = outputs[i].cpu().numpy().transpose(1, 2, 0)
                # multichannel=True와 channel_axis=2를 사용하여 컬러 이미지 SSIM 계산
                score = ssim(img_orig, img_recon, data_range=1.0, multichannel=True, channel_axis=2)
                ssim_scores.append(score)
    return np.mean(ssim_scores)  # 평균 SSIM 반환

# --- 이미지 시각화 함수 (컬러 이미지용) ---
def visualize_reconstruction(model, test_loader, wave_function, device, num_images=5):
    """
    원본 이미지와 모델이 재구성한 이미지를 시각화합니다.
    """
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 기울기 계산 비활성화
        images, _ = next(iter(test_loader))  # 테스트 로더에서 첫 번째 배치 가져오기
        images = images.to(device)
        reconstructed = model(images[:num_images])  # 지정된 수의 이미지 재구성

        plt.figure(figsize=(10, 4))
        for i in range(num_images):
            # 원본 이미지 표시
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0))  # [C, H, W] -> [H, W, C]
            plt.title('Original')
            plt.axis('off')

            # 재구성된 이미지 표시
            plt.subplot(2, num_images, i + num_images + 1)
            plt.imshow(reconstructed[i].cpu().numpy().transpose(1, 2, 0))
            plt.title('Reconstructed')
            plt.axis('off')
        plt.suptitle(f'Reconstruction using {wave_function} wave function', fontsize=16)  # 전체 제목
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 제목 공간 확보
        plt.show()

# --- 학습 및 평가 함수 ---
def train_and_evaluate(optimizer_class, optimizer_name, wave_function='sin', num_epochs=20):
    """
    주어진 옵티마이저와 파형 함수를 사용하여 모델을 학습하고 평가합니다.
    """
    model = ConvAutoencoder().to(device)
    # 옵티마이저 초기화. betas 튜플 언패킹을 사용
    optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=1e-4, wave_function=wave_function)
    criterion = nn.MSELoss()  # 재구성 손실 함수로 MSE 사용
    print(f" --- {optimizer_name} 학습 시작 (SVHN 오토인코더, 파형: {wave_function}) ---")

    train_losses = []
    test_losses = []
    ssim_scores = []
    resonance_means = []  # resonance 값 평균 저장을 위한 리스트

    for epoch in range(1, num_epochs + 1):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0
        current_epoch_resonance_sum = 0.0
        batches_processed_with_res = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()  # 이전 기울기 초기화

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()  # 역전파

            # resonance 값 수집
            batch_res_sum = 0.0
            param_count_for_res = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'resonance' in state and state['resonance'] is not None:
                        # resonance 텐서가 비어있지 않은지 확인
                        if state['resonance'].numel() > 0:
                            batch_res_sum += state['resonance'].abs().mean().item()
                            param_count_for_res += 1
            if param_count_for_res > 0:
                current_epoch_resonance_sum += batch_res_sum / param_count_for_res
            batches_processed_with_res += 1

            optimizer.step()  # 파라미터 업데이트
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 테스트 세트 평가
        model.eval()  # 모델을 평가 모드로 설정
        test_loss = 0.0
        with torch.no_grad():  # 기울기 계산 비활성화
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, inputs).item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # SSIM 계산
        avg_ssim = calculate_ssim(model, test_loader, device)
        ssim_scores.append(avg_ssim)

        # 에포크별 resonance 평균 계산
        avg_resonance_for_epoch = current_epoch_resonance_sum / batches_processed_with_res if batches_processed_with_res > 0 else 0
        resonance_means.append(avg_resonance_for_epoch)

        print(f"Epoch {epoch:2d}/{num_epochs}: Train Loss={avg_train_loss:.6f}, Test Loss={avg_test_loss:.6f}, SSIM={avg_ssim:.4f}, Resonance_Mean={avg_resonance_for_epoch:.6f}")

    # 학습 완료 후 재구성 이미지 시각화
    visualize_reconstruction(model, test_loader, wave_function, device)

    return train_losses, test_losses, ssim_scores, resonance_means

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    print(" === 오토인코더 학습 시작: FWROptimizer (sin 파형) ===")
    # FWROptimizer (sin 파형)으로 학습 및 평가
    train_l, test_l, ssim_s, res_m = train_and_evaluate(
        FWROptimizer, "FWROptimizer", wave_function='sin', num_epochs=20
    )

    print(f" 최종 Test Loss: {test_l[-1]:.6f}, 최종 SSIM: {ssim_s[-1]:.4f}")

    # 손실 및 SSIM 곡선 플롯
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_l) + 1), train_l, label='Train Loss')
    plt.plot(range(1, len(test_l) + 1), test_l, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (FWROptimizer with Sin Wave)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ssim_s) + 1), ssim_s, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Curve (FWROptimizer with Sin Wave)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Resonance Mean 곡선 플롯 추가
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(res_m) + 1), res_m, label='Average Resonance Magnitude')
    plt.xlabel('Epoch')
    plt.ylabel('Average Resonance Magnitude')
    plt.title('Average Resonance Magnitude per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(" === 다른 파형 함수 테스트 === ")

    # 다른 파형 함수로도 실험해볼 수 있도록 예시 추가 (선택 사항)
    # 예를 들어, 'cos' 파형으로 실행하려면 아래 주석을 해제해봐.
    # print(" === 오토인코더 학습 시작: FWROptimizer (cos 파형) ===")
    # train_l_cos, test_l_cos, ssim_s_cos, res_m_cos = train_and_evaluate(
    # FWROptimizer, "FWROptimizer", wave_function='cos', num_epochs=20
    # )
    # print(f" 최종 Test Loss (cos): {test_l_cos[-1]:.6f}, 최종 SSIM (cos): {ssim_s_cos[-1]:.4f}")

    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(train_l_cos) + 1), train_l_cos, label='Train Loss (cos)')
    # plt.plot(range(1, len(test_l_cos) + 1), test_l_cos, label='Test Loss (cos)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve (FWROptimizer with Cos Wave)')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, len(ssim_s_cos) + 1), ssim_s_cos, label='SSIM (cos)')
    # plt.xlabel('Epoch')
    # plt.ylabel('SSIM')
    # plt.title('SSIM Curve (FWROptimizer with Cos Wave)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
