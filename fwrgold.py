# FWR Optimizer with Full Metric Logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Dataset
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# FWR Optimizer
class FWROptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, wave_function='sin'):
        defaults = dict(lr=lr, betas=betas, eps=eps, wave_function=wave_function)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['flow'] = torch.zeros_like(p.data)
                    state['wave'] = torch.zeros_like(p.data)
                    state['res'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)
                state['step'] += 1
                flow = state['flow']
                wave = state['wave']
                res = state['res']
                prev = state['prev_grad']
                delta = grad - prev

                flow.mul_(beta1).add_(grad, alpha=1 - beta1)
                wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                res_update = torch.sin(delta)
                res.mul_(beta3).add_(res_update, alpha=1 - beta3)

                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                bc3 = 1 - beta3 ** state['step']
                m_hat = flow / bc1
                v_hat = wave / bc2
                r_hat = res / bc3

                update = (m_hat + r_hat) / (v_hat.sqrt() + eps)
                p.data.add_(-group['lr'] * update)
                prev.copy_(grad)

# 평가 함수
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# 학습 함수
def train_fwr(epochs=10):
    model = SimpleCNN().to(device)
    optimizer = FWROptimizer(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    resonance_means = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total, res_sum, res_count = 0, 0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'res' in state:
                        res_sum += state['res'].abs().mean().item()
                        res_count += 1

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        res_mean = res_sum / res_count if res_count else 0

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        resonance_means.append(res_mean)

        print(f"Epoch {epoch+1:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Resonance: {res_mean:.5f}")

    return train_losses, test_losses, train_accs, test_accs, resonance_means

# 실행
train_l, test_l, train_a, test_a, res = train_fwr(epochs=10)

# 시각화
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(train_l, label='Train Loss')
plt.plot(test_l, label='Test Loss')
plt.legend(); plt.title("Loss Curve"); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_a, label='Train Acc')
plt.plot(test_a, label='Test Acc')
plt.legend(); plt.title("Accuracy Curve"); plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(res, label='Resonance Mean')
plt.title("Resonance Mean over Epochs")
plt.grid(True)
plt.legend()
plt.show()
