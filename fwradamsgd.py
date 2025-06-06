#FWRADAM,SGD 등 결합예제
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models 

# --- 옵티마이저 공통 로직을 위한 베이스 클래스 ---
class FWROptimizerBase(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=0):
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

        # Update biased first moment estimate (flow)
        flow.mul_(beta1).add_(grad, alpha=1 - beta1) # In-place operations are good for memory
        # Update biased second raw moment estimate (wave)
        wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        # Update biased resonance estimate (grad_delta)
        grad_delta = grad - prev_grad
        resonance.mul_(beta3).add_(grad_delta, alpha=1 - beta3)

    def _get_biased_corrected(self, state, step, beta1, beta2, beta3):
        # Bias correction
        flow_hat = state['flow'] / (1 - beta1**step)
        wave_hat = state['wave'] / (1 - beta2**step)
        resonance_hat = state['resonance'] / (1 - beta3**step)
        return flow_hat, wave_hat, resonance_hat

    # 기본 업데이트 로직: FWRAdam과 FWROptimizer가 공유
    def _apply_update(self, p, group, flow_hat, wave_hat, resonance_hat):
        denom = wave_hat.sqrt().add_(group.get('eps', 1e-8)) # eps 기본값 적용
        update = (flow_hat + resonance_hat) / denom
        p.data.add_(-group['lr'], update)

    @torch.no_grad() # 이 함수 내에서는 gradient 계산이 필요 없음을 명시
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): # closure가 grad 계산을 필요로 할 수 있으므로 enable
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data # .data 접근은 권장되지 않으나, 옵티마이저 구현에서는 일반적
                state = self.state[p]

                if len(state) == 0:
                    self._init_state(p)

                state['step'] += 1
                step = state['step']

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay']) # L2 regularization effect

                self._update_moments(grad, state, beta1, beta2, beta3)

                flow_hat, wave_hat, resonance_hat = self._get_biased_corrected(state, step, beta1, beta2, beta3)

                self._apply_update(p, group, flow_hat, wave_hat, resonance_hat)

                state['prev_grad'].copy_(grad) # 다음 스텝을 위해 현재 grad 저장
        return loss

# FWRAdam 구현 (FWROptimizerBase 상속 및 Adam 특화 로직 추가)
class FWRAdam(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, weight_decay)
        for group in self.param_groups:
            group.setdefault('eps', eps) # Adam에만 필요한 'eps' 기본값 설정

# FWROptimizer 구현 (FWROptimizerBase 상속 및 단순화된 로직)
class FWROptimizer(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.999, 0.9), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, weight_decay)
        for group in self.param_groups:
            group.setdefault('eps', eps) # FWROptimizer에만 필요한 'eps' 기본값 설정

# FWRSGD 구현 (FWROptimizerBase 상속 및 SGD 특화 로직)
class FWRSGD(FWROptimizerBase):
    def __init__(self, params, lr=0.001, betas=(0.95, 0.9, 0.9), weight_decay=5e-4):
        super().__init__(params, lr, betas, weight_decay)

    # SGD 스타일 업데이트는 denom (wave_hat)이 없으므로 오버라이드
    def _apply_update(self, p, group, flow_hat, wave_hat, resonance_hat):
        update = flow_hat + resonance_hat
        p.data.add_(-group['lr'], update)

# --- 데이터셋 및 DataLoader (CIFAR-100으로 변경) ---
# CIFAR-100 데이터셋의 평균과 표준편차
CIFAR100_MEAN = (0.5070751592, 0.4865488733, 0.4409178433)
CIFAR100_STD = (0.2673342858, 0.2564384629, 0.2761504713)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 데이터 증강
    transforms.RandomHorizontalFlip(), # 데이터 증강
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD) # 정규화
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

# --- CIFAR-100 데이터셋 로드 ---
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)

# DataLoader 설정
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 학습 및 평가 함수 ---
def train(optimizer_class, optimizer_name):
    # 모델을 ResNet18로 변경 (사전 학습되지 않은 모델, 100개 클래스)
    model = models.resnet18(weights=None, num_classes=100).to(device)

    # 옵티마이저 인스턴스 생성 (ResNet18은 weight_decay를 5e-4로 설정하는 경우가 많음)
    if optimizer_name == "FWRSGD":
        optimizer = optimizer_class(model.parameters(), lr=0.01, betas=(0.95, 0.9, 0.9), weight_decay=5e-4)
    elif optimizer_name == "Adam (default)":
        optimizer = optimizer_class(model.parameters(), lr=0.001, weight_decay=5e-4)
    elif optimizer_name == "SGD (default)":
        optimizer = optimizer_class(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # SGD에 momentum 추가
    else: # FWROptimizer, FWRAdam
        optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.95, 0.999, 0.9), weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    print(f"\n--- {optimizer_name} Training (ResNet18 on CIFAR-100) ---")

    num_epochs = 100 # 에포크 수를 100으로 변경

    # 옵티마이저가 resonance를 사용하는지 미리 확인
    is_resonance_optimizer = optimizer_name in ["FWROptimizer", "FWRAdam", "FWRSGD"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_resonance_sum = 0.0
        num_resonance_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # resonance 계산 로직 개선
            if is_resonance_optimizer:
                batch_resonance_total = 0.0
                current_batch_params_with_resonance = 0
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        # 'resonance' 키가 있고, 값이 None이 아닌 경우에만 평균에 포함
                        if 'resonance' in state and state['resonance'] is not None:
                            batch_resonance_total += state['resonance'].abs().mean().item()
                            current_batch_params_with_resonance += 1
                if current_batch_params_with_resonance > 0:
                    epoch_resonance_sum += batch_resonance_total / current_batch_params_with_resonance
                    num_resonance_batches += 1

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f" Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Resonance 출력 부분 수정
        if is_resonance_optimizer and num_resonance_batches > 0:
            avg_resonance_for_epoch = epoch_resonance_sum / num_resonance_batches
            print(f"Epoch {epoch}/{num_epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, Resonance={avg_resonance_for_epoch:.6f}")
        else:
            print(f"Epoch {epoch}/{num_epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%, Resonance=N/A")


    # --- 테스트 단계 ---
    model.eval() # 모델을 평가 모드로 전환
    correct = 0
    total = 0
    with torch.no_grad(): # gradient 계산을 비활성화
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100. * correct / total
    print(f"{optimizer_name} Test Accuracy: {test_acc:.2f}%\n")

# --- 메인 실행 블록 (num_workers 사용 시 필요) ---
if __name__ == '__main__':
    # 옵티마이저별 학습 실행
    # FWROptimizer와 Adam (default) 비교
    train(FWROptimizer, "FWROptimizer")
    #train(torch.optim.Adam, "Adam (default)")

    # 다른 옵티마이저도 테스트하려면 주석 해제
    # train(FWRAdam, "FWRAdam")
    # train(FWRSGD, "FWRSGD")
    # train(torch.optim.SGD, "SGD (default)")
