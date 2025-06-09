# 구글 코랩용: FWR 옵티마이저로 비선형 회귀 + 황금비 적용
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

ϕ = (1 + 5 ** 0.5) / 2  # 황금비

# ▶ 1. 데이터 생성 (sin + noise)
N = 256
x = torch.linspace(-3, 3, N).view(-1, 1)
y = torch.sin(2 * x) + 0.3 * torch.randn_like(x)

# ▶ 2. 단순한 회귀 모델
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# ▶ 3. FWR Optimizer 정의 (황금비 포함)
class FWROptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.95, 0.999, 0.9), eps=1e-8, wave_function='sin'):
        defaults = dict(lr=lr, betas=betas, eps=eps, wave_function=wave_function)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['flow'] = torch.zeros_like(p)
                    state['wave'] = torch.zeros_like(p)
                    state['res'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    state['step'] = 0

                flow, wave, res = state['flow'], state['wave'], state['res']
                prev_grad = state['prev_grad']
                beta1, beta2, beta3 = group['betas']
                lr, eps = group['lr'], group['eps']

                state['step'] += 1

                flow.mul_(beta1).add_(grad, alpha=1 - beta1)
                wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                delta = grad - prev_grad

                if group['wave_function'] == 'sin':
                    res_update = torch.sin(delta)
                elif group['wave_function'] == 'tanh':
                    res_update = torch.tanh(delta)
                else:
                    res_update = delta

                res.mul_(beta3).add_(res_update, alpha=1 - beta3)

                # 바이어스 보정
                step = state['step']
                flow_hat = flow / (1 - beta1 ** step)
                wave_hat = wave / (1 - beta2 ** step)
                res_hat = res / (1 - beta3 ** step)

                denom = wave_hat.sqrt().add_(eps)
                update = (flow_hat + res_hat * ϕ) / denom

                p.add_(-lr * update)
                state['prev_grad'].copy_(grad)

# ▶ 4. 학습
criterion = nn.MSELoss()
optimizer = FWROptimizer(model.parameters(), lr=0.01)

epochs = 300
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ▶ 5. 결과 시각화
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(x.numpy(), y.numpy(), 'b.', label='Data')
plt.plot(x.detach().numpy(), model(x).detach().numpy(), 'r-', label='FWR fit')
plt.title('FWR with Golden Ratio')
plt.legend()

plt.subplot(1,2,2)
plt.plot(losses)
plt.title('Loss over Epochs')
plt.grid(True)
plt.show()
