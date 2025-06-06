#Convolutional Autoencoder 예제
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- Sawtooth 함수 정의 (사용 안 하지만 유지) ---
def sawtooth(x, period=1.0):
    return 2 * (x / period - torch.floor(0.5 + x / period))

# --- FWR Optimizer Base ---
class FWROptimizerBase(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=0, wave_function='sin'):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, wave_function=wave_function)
        super().__init__(params, defaults)
        valid_functions = ['sin', 'cos', 'tanh', 'sawtooth']
        if wave_function not in valid_functions:
            raise ValueError(f"wave_function must be one of {valid_functions}, got {wave_function}")

    def _init_state(self, p):
        state = self.state[p]
        state['step'] = 0
        state['flow'] = torch.zeros_like(p.data)
        state['wave'] = torch.zeros_like(p.data)
        state['resonance'] = torch.zeros_like(p.data)
        state['prev_grad'] = torch.zeros_like(p.data)

    def _update_moments(self, grad, state, beta1, beta2, beta3, wave_function):
        flow = state['flow']
        wave = state['wave']
        resonance = state['resonance']
        prev_grad = state['prev_grad']
        flow.mul_(beta1).add_(grad, alpha=1 - beta1)
        wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        grad_delta = grad - prev_grad
        if wave_function == 'sin':
            resonance_update = torch.sin(grad_delta)
        elif wave_function == 'cos':
            resonance_update = torch.cos(grad_delta)
        elif wave_function == 'tanh':
            resonance_update = torch.tanh(grad_delta)
        elif wave_function == 'sawtooth':
            resonance_update = sawtooth(grad_delta)
        resonance.mul_(beta3).add_(resonance_update, alpha=1 - beta3)

    def _get_biased_corrected(self, state, step, beta1, beta2, beta3):
        flow_hat = state['flow'] / (1 - beta1**step)
        wave_hat = state['wave'] / (1 - beta2**step)
        resonance_hat = state['resonance'] / (1 - beta3**step)
        return flow_hat, wave_hat, resonance_hat

    def _apply_update(self, p, group, flow_hat, wave_hat, resonance_hat):
        denom = wave_hat.sqrt().add_(group.get('eps', 1e-8))
        update = (flow_hat + resonance_hat) / denom
        p.data.add_(-group['lr'], update)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            wave_function = group['wave_function']
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
                self._update_moments(grad, state, beta1, beta2, beta3, wave_function)
                flow_hat, wave_hat, resonance_hat = self._get_biased_corrected(state, step, beta1, beta2, beta3)
                self._apply_update(p, group, flow_hat, wave_hat, resonance_hat)
                state['prev_grad'].copy_(grad)
        return loss

# --- FWR Optimizer ---
class FWROptimizer(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, weight_decay=0, wave_function='sin'):
        super().__init__(params, lr, betas, weight_decay, wave_function)
        for group in self.param_groups:
            group.setdefault('eps', eps)

# --- Convolutional Autoencoder 모델 정의 ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), # [1, 28, 28] -> [16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [16, 28, 28] -> [16, 14, 14]
            nn.Conv2d(16, 8, 3, padding=1), # [16, 14, 14] -> [8, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # [8, 14, 14] -> [8, 7, 7]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), # [8, 7, 7] -> [16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), # [16, 14, 14] -> [8, 28, 28]
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1), # [8, 28, 28] -> [1, 28, 28]
            nn.Sigmoid() # 출력 범위 [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 데이터 로딩 ---
transform = transforms.Compose([
    transforms.ToTensor(), # [0, 1] 범위로 변환
])

train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- SSIM 계산 함수 ---
def calculate_ssim(model, test_loader, device):
    model.eval()
    ssim_scores = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i in range(inputs.size(0)):
                img_orig = inputs[i].cpu().numpy().squeeze()
                img_recon = outputs[i].cpu().numpy().squeeze()
                score = ssim(img_orig, img_recon, data_range=1.0)
                ssim_scores.append(score)
    return np.mean(ssim_scores)

# --- 이미지 시각화 함수 ---
def visualize_reconstruction(model, test_loader, wave_function, device, num_images=5):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        reconstructed = model(images[:num_images])
        
        plt.figure(figsize=(10, 4))
        for i in range(num_images):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(2, num_images, i + num_images + 1)
            plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')
        plt.suptitle(f'Reconstruction with {wave_function}')
        plt.tight_layout()
        plt.show()

# --- 학습 및 평가 함수 ---
def train_and_evaluate(optimizer_class, optimizer_name, wave_function='sin'):
    model = ConvAutoencoder().to(device)
    optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=1e-4, wave_function=wave_function)
    criterion = nn.MSELoss()
    print(f"\n--- {optimizer_name} Training (ConvAutoencoder on FashionMNIST, Wave Function: {wave_function}) ---")

    num_epochs = 20 # 20 에포크로 변경
    train_losses = []
    test_losses = []
    ssim_scores = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        resonance_sum = 0.0
        batches_with_res = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()

            res_sum = 0.0
            n_params = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'resonance' in state:
                        res_sum += state['resonance'].abs().mean().item()
                        n_params += 1
            if n_params > 0:
                resonance_sum += res_sum / n_params
                batches_with_res += 1

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, inputs).item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        avg_ssim = calculate_ssim(model, test_loader, device)
        ssim_scores.append(avg_ssim)

        avg_resonance = resonance_sum / batches_with_res if batches_with_res > 0 else 0
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Test Loss={avg_test_loss:.6f}, SSIM={avg_ssim:.4f}, Resonance={avg_resonance:.6f}")

    visualize_reconstruction(model, test_loader, wave_function, device)

    return train_losses, test_losses, ssim_scores

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    print("\n=== Training with Wave Function: sin ===")
    train_losses, test_losses, ssim_scores = train_and_evaluate(FWROptimizer, "FWROptimizer", wave_function='sin')
    print(f"Final Test Loss: {test_losses[-1]:.6f}, Final SSIM: {ssim_scores[-1]:.4f}")

    # 손실 및 SSIM 곡선 플롯
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves (sin)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ssim_scores) + 1), ssim_scores, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Curve (sin)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
