
"""
FWR Adaptive Training Framework - Ablation Study
CIFAR-10 + CNN (경량)

비교군:
1. Baseline: 일반 CrossEntropy 학습
2. +FWR: FWR 안정성 제어기만 적용
3. +FWR + AutoTarget: FWR + Autonomous Target Setter
4. +FWR + Target + SafeMode: 전체 적용

측정: Test Accuracy, Convergence Speed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time


# ============================================================
# CNN (CIFAR-10)
# ============================================================
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ============================================================
# 데이터
# ============================================================
def get_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size, shuffle=False, drop_last=True, num_workers=2)
    return train_loader, test_loader


# ============================================================
# FWR Compact
# ============================================================
class FWRController(nn.Module):
    def __init__(self, r_max=8.0, rq_threshold=0.3):
        super().__init__()
        self.r_max = r_max
        self.rq_threshold = rq_threshold
        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.register_buffer('r_history', torch.zeros(10))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('init', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('safe_ema', torch.zeros(1))
        self._pending = None

    @property
    def safety_margin(self):
        return F_pt.softplus(self.raw_safety_margin) + 0.05

    def forward(self, f_t, w_t, r_t):
        prev_ptr = (self.ptr - 1) % 10
        r_prev = self.r_history[prev_ptr]
        v = r_t - r_prev
        damping = torch.exp(-0.5 * (1.0 + torch.abs(v)) * F_pt.relu(r_t - self.r_max))
        r_adj = r_t * damping
        e_t = f_t * w_t * r_adj
        
        r_std = torch.std(r_t) + 1e-8 if r_t.numel() > 1 else torch.tensor(1e-8)
        r_excess = torch.mean(F_pt.relu(r_t - self.r_max))
        r_bal = F_pt.relu(self.r_max * 0.5 - torch.mean(r_t)) / (self.r_max * 0.5 + 1e-6)
        rq = torch.exp(-(r_std + r_excess + 0.1 * r_bal))
        
        safe = False
        sf = torch.sigmoid((self.rq_threshold - rq) / self.safety_margin)
        if rq < self.rq_threshold:
            safe = True
            f_s = f_t * (1.0 - 0.5 * sf)
            w_s = self.safe_w_base.expand_as(w_t) * sf + w_t * (1.0 - sf)
            e_t = f_s * w_s * r_adj
        
        self._pending = (safe, r_t.detach().mean())
        return e_t, r_adj, rq, torch.mean(e_t), safe

    def commit_state(self):
        if self._pending:
            safe, rm = self._pending
            p = self.ptr.item()
            self.r_history[p] = rm
            self.ptr[0] = (p + 1) % 10
            self.init[0] = True
            self.safe_ema = 0.99 * self.safe_ema + 0.01 * (1.0 if safe else 0.0)
            self._pending = None

    def get_auxiliary_loss(self, rm=None, rp=None):
        return torch.exp(-self.safety_margin * 10.0) + self.safe_ema * 0.5


# ============================================================
# 실험 실행
# ============================================================
def run_experiment(name, model, train_loader, test_loader, epochs=30,
                   use_fwr=False, use_target=False, use_safemode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    if use_fwr:
        fwr = FWRController(r_max=8.0, rq_threshold=0.3).to(device)
        optimizer.add_param_group({'params': fwr.parameters()})
        target = 1.0
        perf_ema = 1.0
    
    history = {'epoch': [], 'train_loss': [], 'test_acc': [], 'time': []}
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_loss = criterion(logits, y)
            loss = ce_loss
            
            if use_fwr:
                with torch.no_grad():
                    probs = F_pt.softmax(logits, dim=1)
                    f_t = probs.max(1)[0].unsqueeze(1)
                    w_t = torch.sigmoid(torch.randn(x.size(0), 1, device=device))
                    loss_per_sample = F_pt.cross_entropy(logits, y, reduction='none')
                    r_t = 8.0 * torch.exp(-0.4 * loss_per_sample.unsqueeze(1) / 2.0)
                
                e_out, r_adj, rq, perf, safe = fwr(f_t, w_t, r_t)
                
                if use_target:
                    perf_ema = 0.9 * perf_ema + 0.1 * perf.item()
                    if rq.item() > 0.6: mult = 1.30
                    elif rq.item() > 0.3: mult = 1.15
                    else: mult = 1.05
                    target = max(perf_ema * mult, 0.5)
                    loss = loss + 0.05 * F_pt.mse_loss(e_out, torch.ones_like(e_out) * target)
                
                if use_safemode and safe:
                    loss = loss + 0.01 * torch.mean(f_t)
                
                loss = loss + 0.01 * fwr.get_auxiliary_loss()
                fwr.commit_state()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += ce_loss.item()
            n += 1
        
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss / max(n, 1))
        history['test_acc'].append(100.0 * correct / total)
        history['time'].append(time.time() - t0)
    
    return history


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    train_loader, test_loader = get_cifar10(batch_size=128)
    print(f"CIFAR-10 CNN | Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    configs = [
        ("1. Baseline",           False, False, False),
        ("2. +FWR",               True,  False, False),
        ("3. +FWR+AutoTarget",    True,  True,  False),
        ("4. +FWR+Target+Safe",   True,  True,  True),
    ]
    
    results = {}
    
    for name, use_fwr, use_target, use_safemode in configs:
        print(f"\n{'='*60}")
        print(f"실험: {name}")
        print(f"{'='*60}")
        
        model = CNN(num_classes=10)
        history = run_experiment(name, model, train_loader, test_loader, epochs=30,
                                 use_fwr=use_fwr, use_target=use_target, use_safemode=use_safemode)
        results[name] = history
        
        print(f"  최종 Acc: {history['test_acc'][-1]:.2f}% | "
              f"최고 Acc: {max(history['test_acc']):.2f}% | "
              f"시간: {sum(history['time'])/len(history['time']):.1f}s/epoch")
    
    # 최종 비교
    print(f"\n{'='*65}")
    print("Ablation Study - CIFAR-10 + CNN (30에포크)")
    print(f"{'='*65}")
    print(f"{'설정':<30} {'최종Acc':<10} {'최고Acc':<10} {'시간':<10}")
    print("-"*60)
    for name in results:
        h = results[name]
        print(f"{name:<30} {h['test_acc'][-1]:<10.2f} {max(h['test_acc']):<10.2f} "
              f"{sum(h['time'])/len(h['time']):<10.1f}")
