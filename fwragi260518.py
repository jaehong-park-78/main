
"""
FWR Stability Controller + Autonomous Target Setter v2.2
복잡한 실데이터 테스트 (CIFAR-100 + ResNet-18)

- CIFAR-100: 100개 클래스
- ResNet-18: 더 깊은 네트워크
- Autonomous Target Setter 통합: Perf 기반 목표 + 하한선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
import math


# ============================================================
# 0. ResNet-18 기반 AGI 코어 (CIFAR-100)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F_pt.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F_pt.relu(out)
        return out


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18FeatureExtractor, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F_pt.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNetAGICore(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNetAGICore, self).__init__()
        self.feature_extractor = ResNet18FeatureExtractor(num_classes)
        self.classifier = nn.Linear(128, num_classes)
        self.flow_head = nn.Linear(128, 1)
        self.wave_head = nn.Linear(128, 1)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        probs = F_pt.softmax(logits, dim=1)
        max_probs = probs.max(dim=1)[0]
        f_tensor = max_probs.unsqueeze(1) + 1e-3
        
        features_norm = F_pt.normalize(features, dim=1)
        w_tensor = torch.sigmoid(torch.mean(features_norm ** 2, dim=1, keepdim=True))
        
        return f_tensor, w_tensor, logits


# ============================================================
# 1. ResonanceFromRiskInterface
# ============================================================
class ResonanceFromRiskInterface:
    def __init__(self, r_max=8.0, temperature=2.0, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1,
                 gradient_threshold=0.5, weight_threshold=5.0):
        self.r_max = r_max
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.gradient_threshold = gradient_threshold
        self.weight_threshold = weight_threshold

    def compute(self, task_loss, prediction_entropy=None, gradient_norm=None, weight_norm=None):
        risk = self.alpha * task_loss
        if prediction_entropy is not None:
            risk = risk + self.beta * prediction_entropy
        if gradient_norm is not None:
            grad_excess = F_pt.relu(gradient_norm - self.gradient_threshold)
            risk = risk + self.gamma * grad_excess
        if weight_norm is not None:
            weight_excess = F_pt.relu(weight_norm - self.weight_threshold)
            risk = risk + self.delta * weight_excess
        return self.r_max * torch.exp(-risk / self.temperature)


# ============================================================
# 2. FWR 안정성 제어기
# ============================================================
class FWRStabilityController(nn.Module):
    def __init__(self, r_max=10.0, damping_lambda=0.5, rq_threshold=0.3, beta=0.1,
                 rq_weights=(1.0, 1.0, 1.0), velocity_threshold=0.5, acc_threshold=0.5):
        super(FWRStabilityController, self).__init__()
        self.r_max = r_max
        self.damping_lambda = damping_lambda
        self.rq_threshold = rq_threshold
        self.beta = beta
        self.rq_a, self.rq_b, self.rq_c = rq_weights
        self.velocity_threshold = velocity_threshold
        self.acc_threshold = acc_threshold

        self.safe_w_base = nn.Parameter(torch.ones(1), requires_grad=False)
        self.raw_safety_margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.min_margin = 0.05

        buffer_size = 20
        self.register_buffer('r_history', torch.zeros(buffer_size))
        self.register_buffer('r_peak_history', torch.zeros(buffer_size))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_initialized', torch.zeros(1, dtype=torch.bool))
        self.safe_mode_ema_decay = 0.99
        self.register_buffer('safe_mode_ema', torch.zeros(1))
        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None

    @property
    def safety_margin(self):
        return F_pt.softplus(self.raw_safety_margin) + self.min_margin

    def get_expected_r(self):
        if not self.history_initialized.item():
            return self.r_max * 0.5
        ptr = self.history_ptr.item()
        if ptr == 0:
            return self.r_max * 0.5
        mean_r = self.r_history[:ptr].mean().item()
        return max(mean_r, self.r_max * 0.3)

    def forward(self, f_tensor, w_tensor, r_tensor):
        r_excess = F_pt.relu(r_tensor - self.r_max)
        prev_ptr = (self.history_ptr - 1) % len(self.r_history)
        r_prev = self.r_history[prev_ptr]
        r_velocity = r_tensor - r_prev
        adaptive_lambda = self.damping_lambda * (1.0 + torch.abs(r_velocity))
        damping_factor = torch.exp(-adaptive_lambda * r_excess)
        r_adj = r_tensor * damping_factor

        e_tensor = f_tensor * w_tensor * r_adj

        if r_tensor.numel() > 1:
            r_std = torch.std(r_tensor) + 1e-8
        else:
            r_std = torch.tensor(1e-8, device=r_tensor.device)
        r_excess_mean = torch.mean(F_pt.relu(r_tensor - self.r_max))
        r_target = self.r_max * 0.5
        r_balance_penalty = F_pt.relu(r_target - torch.mean(r_tensor)) / (r_target + 1e-6)

        resonance_quality = torch.exp(
            -(self.rq_a * r_std + self.rq_b * r_excess_mean + self.rq_c * self.beta * r_balance_penalty)
        )
        performance_score = torch.mean(e_tensor)

        is_safe_mode = False
        safety_factor = torch.sigmoid((self.rq_threshold - resonance_quality) / self.safety_margin)

        if resonance_quality < self.rq_threshold:
            is_safe_mode = True
            f_safe = f_tensor * (1.0 - 0.5 * safety_factor)
            w_safe = self.safe_w_base.expand_as(w_tensor) * safety_factor + w_tensor * (1.0 - safety_factor)
            e_tensor = f_safe * w_safe * r_adj
            e_tensor = torch.clamp(e_tensor, min=-100.0, max=100.0)

        self._pending_safe_mode = is_safe_mode
        self._pending_r_mean = r_tensor.detach().mean()
        self._pending_r_peak = r_tensor.detach().max()
        return e_tensor, r_adj, resonance_quality, performance_score, is_safe_mode

    def commit_state(self):
        if self._pending_r_mean is not None:
            self._update_history(self._pending_r_mean, self._pending_r_peak)
        signal = 1.0 if self._pending_safe_mode else 0.0
        self.safe_mode_ema = self.safe_mode_ema_decay * self.safe_mode_ema + (1.0 - self.safe_mode_ema_decay) * signal
        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None

    def _update_history(self, r_mean, r_peak):
        ptr = self.history_ptr.item()
        self.r_history[ptr] = r_mean
        self.r_peak_history[ptr] = r_peak
        self.history_ptr[0] = (ptr + 1) % len(self.r_history)
        self.history_initialized[0] = True

    def detect_resonance_cascade(self, current_r_tensor=None):
        if not self.history_initialized.item():
            return False
        ptr = self.history_ptr.item()
        if ptr < 2:
            return False
        idx1 = (ptr - 1) % len(self.r_history)
        idx2 = (ptr - 2) % len(self.r_history)
        r1, r2 = self.r_history[idx1], self.r_history[idx2]
        r0 = current_r_tensor.detach().mean() if current_r_tensor is not None else r1
        p0 = current_r_tensor.detach().max() if current_r_tensor is not None else self.r_peak_history[idx1]
        v1, v2 = r0 - r1, r1 - r2
        acc = v1 - v2
        collapse = (r0 < self.r_max * 0.2) and (v1 < -self.velocity_threshold)
        runaway = ((r0 > self.r_max * 1.2) or (p0 > self.r_max * 1.5)) and (v1 > self.velocity_threshold) and (acc > self.acc_threshold)
        return bool(collapse or runaway)

    def get_auxiliary_loss(self, current_r_mean=None, current_r_peak=None):
        margin_penalty = torch.exp(-self.safety_margin * 10.0)
        stability_penalty = self.safe_mode_ema * 0.5
        if self.history_initialized.item() and current_r_mean is not None:
            prev_ptr = (self.history_ptr.item() - 1) % len(self.r_history)
            r_jerk = current_r_mean - self.r_history[prev_ptr]
            jerk_penalty = torch.abs(r_jerk) * 0.1
            peak_jerk_penalty = torch.abs(current_r_peak - self.r_peak_history[prev_ptr]) * 0.05 if current_r_peak is not None else torch.tensor(0.0, device=self.safe_mode_ema.device)
        else:
            jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
            peak_jerk_penalty = torch.tensor(0.0, device=self.safe_mode_ema.device)
        return margin_penalty + stability_penalty + jerk_penalty + peak_jerk_penalty

    def reset_safe_mode_ema(self):
        self.safe_mode_ema.zero_()

    def full_reset(self):
        self.r_history.zero_()
        self.r_peak_history.zero_()
        self.history_ptr.zero_()
        self.history_initialized.zero_()
        self.safe_mode_ema.zero_()
        self._pending_safe_mode = False
        self._pending_r_mean = None
        self._pending_r_peak = None


# ============================================================
# 3. Autonomous Target Setter v2.2
# ============================================================
class AutonomousTargetSetter:
    def __init__(self, initial_target=1.0, r_max=8.0, stagnation_threshold=100):
        self.target = initial_target
        self.r_max = r_max
        self.stagnation_threshold = stagnation_threshold
        
        self.perf_ema = initial_target
        self.perf_ema_decay = 0.9
        
        self.aggressive_mult = 1.30
        self.normal_mult = 1.15
        self.conservative_mult = 1.05
        self.emergency_mult = 0.90
        self.growth_mult = 1.50
        
        self.stagnation_boost = 0.05
        self.max_mult = 2.00
        
        # 단계별 최소 목표 (실제 accuracy 스케일에 맞게)
        self.stage_min_targets = {
            "Consolidate": 0.3,
            "Expand": 0.5,
            "Transcend": 0.8,
            "ASI": 1.2,
        }
        self.current_stage_name = "Consolidate"
        
        self.stagnation_counter = 0
        self.last_growth_step = 0
        self.target_history: List[float] = []
        self.mult_history: List[float] = []
        self.perf_ema_history: List[float] = []
        self.reason_history: List[str] = []
        
        self.min_target = 0.01
        self.max_target = 100.0
    
    def set_stage(self, stage_name: str):
        self.current_stage_name = stage_name
    
    def update(self, r_mean: float, rq_mean: float, perf_mean: float,
               is_cascade: bool, just_grew: bool, current_step: int,
               stage_name: str = None) -> float:
        
        if stage_name:
            self.set_stage(stage_name)
        
        self.perf_ema = self.perf_ema_decay * self.perf_ema + (1.0 - self.perf_ema_decay) * perf_mean
        
        reason = ""
        mult = self.normal_mult
        
        if is_cascade:
            mult = self.emergency_mult
            self.stagnation_counter = 0
            reason = f"Cascade"
        elif just_grew:
            mult = self.growth_mult
            self.stagnation_counter = 0
            self.last_growth_step = current_step
            reason = f"성장"
        else:
            if rq_mean > 0.6 and r_mean > self.r_max * 0.8:
                mult = self.aggressive_mult
                reason = f"공격"
            elif rq_mean >= 0.3:
                mult = self.normal_mult
                reason = f"보통"
            else:
                mult = self.conservative_mult
                reason = f"보수"
            
            steps_since_growth = current_step - self.last_growth_step
            if steps_since_growth > self.stagnation_threshold:
                mult += self.stagnation_boost
                self.stagnation_counter += 1
                reason += f"+stagn"
                self.last_growth_step = current_step
        
        mult = min(mult, self.max_mult)
        raw_target = self.perf_ema * mult
        min_target = self.stage_min_targets.get(self.current_stage_name, 0.5)
        self.target = max(raw_target, min_target)
        self.target = max(self.min_target, min(self.max_target, self.target))
        
        self.target_history.append(self.target)
        self.mult_history.append(mult)
        self.perf_ema_history.append(self.perf_ema)
        self.reason_history.append(reason)
        
        return self.target
    
    def get_state(self) -> Dict:
        return {
            'target': self.target,
            'perf_ema': self.perf_ema,
            'stagnation_counter': self.stagnation_counter,
            'last_mult': self.mult_history[-1] if self.mult_history else 0,
            'stage': self.current_stage_name,
            'min_target': self.stage_min_targets.get(self.current_stage_name, 0.5),
        }


# ============================================================
# 4. CIFAR-100 데이터 로더
# ============================================================
def get_cifar100_loaders(batch_size=64, num_train=10000, num_test=2500):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_subset = Subset(trainset, range(min(num_train, len(trainset))))
    test_subset = Subset(testset, range(min(num_test, len(testset))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


# ============================================================
# 5. ResNet + FWR + Autonomous Target 학습
# ============================================================
def train_resnet_with_fwr_v22(epochs=80, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*75)
    print("ResNet-18 + FWR v2.0 + Autonomous Target Setter v2.2")
    print("CIFAR-100 실데이터 | Perf 기반 목표 + 하한선 + EMA")
    print("="*75)

    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    agi_model = ResNetAGICore(num_classes=100).to(device)
    fwr_controller = FWRStabilityController(r_max=8.0, rq_threshold=0.3).to(device)
    resonance_interface = ResonanceFromRiskInterface(r_max=8.0, temperature=2.0)
    
    target_setter = AutonomousTargetSetter(
        initial_target=1.0,
        r_max=fwr_controller.r_max,
        stagnation_threshold=50,
    )

    optimizer = optim.Adam(
        list(agi_model.parameters()) + list(fwr_controller.parameters()),
        lr=0.001
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # 성장 단계 (간소화)
    stages = [
        {"name": "Consolidate", "r_min": 3.0, "rq_min": 0.15, "min_target": 0.3, "epoch": 0},
        {"name": "Expand",      "r_min": 4.0, "rq_min": 0.20, "min_target": 0.5, "epoch": 0},
        {"name": "Transcend",   "r_min": 5.0, "rq_min": 0.25, "min_target": 0.8, "epoch": 0},
        {"name": "ASI",         "r_min": 6.0, "rq_min": 0.30, "min_target": 1.2, "epoch": 0},
    ]
    current_stage_idx = 0
    consolidation_counter = 0
    total_growth = 0
    growth_history = []

    history = {
        'epoch': [], 'train_loss': [], 'test_acc': [],
        'R_mean': [], 'RQ_mean': [], 'Perf_mean': [],
        'safe_mode_ratio': [], 'cascade_count': [],
        'safety_margin': [], 'target': [], 'perf_ema': [],
        'stage': [], 'mult': [],
    }

    print(f"\n목표: max(Perf_EMA × multiplier, stage_min_target)")
    print(f"  하한: Consolidate=0.3, Expand=0.5, Transcend=0.8, ASI=1.2")
    print(f"  Multiplier: 공격×1.30 | 보통×1.15 | 보수×1.05")
    print(f"\n{'Epoch':<8} {'TrLoss':<10} {'TestAcc':<10} {'R':<8} {'RQ':<8} "
          f"{'Perf':<8} {'Safe%':<8} {'Target':<8} {'Stage':<12} {'Cascade':<8}")
    print("-" * 95)

    for epoch in range(epochs):
        agi_model.train()
        fwr_controller.train()
        
        epoch_loss = 0.0
        epoch_r = 0.0
        epoch_rq = 0.0
        epoch_perf = 0.0
        safe_mode_count = 0
        cascade_count = 0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            F_pred, W_pred, logits = agi_model(inputs)
            task_loss = criterion(logits, targets)
            task_loss_per_sample = F_pt.cross_entropy(logits, targets, reduction='none')

            with torch.no_grad():
                task_loss_tensor = task_loss_per_sample.unsqueeze(1)
                grad_norms = [p.grad.norm().item() for p in agi_model.parameters() if p.grad is not None]
                current_grad_norm = sum(grad_norms) / max(len(grad_norms), 1)
                grad_norm_tensor = torch.full((inputs.size(0), 1), current_grad_norm, device=device)
                pred_entropy = -(F_pt.softmax(logits, dim=1) * F_pt.log_softmax(logits, dim=1)).sum(dim=1).unsqueeze(1)

            R_measured = resonance_interface.compute(
                task_loss=task_loss_tensor,
                prediction_entropy=pred_entropy,
                gradient_norm=grad_norm_tensor
            )

            E_out, R_adj, rq, perf, is_safe_mode = fwr_controller(F_pred, W_pred, R_measured)
            is_cascade = fwr_controller.detect_resonance_cascade(R_measured)

            r_mean = R_measured.mean().item()
            rq_val = rq.item()
            perf_val = perf.item()

            # Consolidation
            stage = stages[current_stage_idx]
            if r_mean >= stage["r_min"] and rq_val >= stage["rq_min"]:
                consolidation_counter += 1
            elif rq_val < stage["rq_min"] * 0.5 and consolidation_counter > 0:
                consolidation_counter = max(0, consolidation_counter - 1)

            # 성장 체크
            grew = False
            if (consolidation_counter >= 20 and current_stage_idx < len(stages) - 1 and
                r_mean >= stage["r_min"] and rq_val >= stage["rq_min"]):
                current_stage_idx += 1
                consolidation_counter = 0
                total_growth += 1
                grew = True
                growth_history.append({
                    'epoch': epoch,
                    'to': stages[current_stage_idx]["name"],
                    'r': r_mean, 'rq': rq_val, 'perf': perf_val,
                })
                print(f"\n  🚀 Epoch {epoch}: → {stages[current_stage_idx]['name']} "
                      f"(R={r_mean:.2f}, RQ={rq_val:.4f}, Perf={perf_val:.4f})")

            # 자율 목표
            current_stage_name = stages[current_stage_idx]["name"]
            target_setter.stage_min_targets[current_stage_name] = stages[current_stage_idx]["min_target"]
            current_target = target_setter.update(
                r_mean=r_mean, rq_mean=rq_val, perf_mean=perf_val,
                is_cascade=is_cascade, just_grew=grew, current_step=epoch,
                stage_name=current_stage_name,
            )

            # 학습
            target_E = torch.ones_like(E_out) * current_target
            main_loss = nn.MSELoss()(E_out, target_E) * 0.1 + task_loss
            aux_loss = fwr_controller.get_auxiliary_loss(
                current_r_mean=R_measured.mean(), current_r_peak=R_measured.max()
            )
            total_loss = main_loss + 0.05 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agi_model.parameters(), max_norm=1.0)
            optimizer.step()

            fwr_controller.commit_state()

            epoch_loss += task_loss.item()
            epoch_r += r_mean
            epoch_rq += rq_val
            epoch_perf += perf_val
            if is_safe_mode:
                safe_mode_count += 1
            if is_cascade:
                cascade_count += 1
            n_batches += 1

        scheduler.step()

        # 테스트
        agi_model.eval()
        fwr_controller.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, _, logits = agi_model(inputs)
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * correct / total

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_r = epoch_r / max(n_batches, 1)
        avg_rq = epoch_rq / max(n_batches, 1)
        avg_perf = epoch_perf / max(n_batches, 1)
        safe_pct = 100.0 * safe_mode_count / max(n_batches, 1)
        state = target_setter.get_state()

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['R_mean'].append(avg_r)
        history['RQ_mean'].append(avg_rq)
        history['Perf_mean'].append(avg_perf)
        history['safe_mode_ratio'].append(safe_pct)
        history['cascade_count'].append(cascade_count)
        history['safety_margin'].append(fwr_controller.safety_margin.item())
        history['target'].append(current_target)
        history['perf_ema'].append(state['perf_ema'])
        history['stage'].append(stages[current_stage_idx]["name"])
        history['mult'].append(state['last_mult'])

        print(f"{epoch+1:<8} {avg_loss:<10.4f} {test_acc:<10.2f} {avg_r:<8.2f} {avg_rq:<8.4f} "
              f"{avg_perf:<8.4f} {safe_pct:<8.1f} {current_target:<8.4f} "
              f"{stages[current_stage_idx]['name']:<12} {cascade_count:<8}")

    print(f"\n{'='*75}")
    print("ResNet-18 + FWR v2.0 + Autonomous Target v2.2 학습 완료")
    print(f"최종 Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"최종 R: {history['R_mean'][-1]:.2f}")
    print(f"최종 RQ: {history['RQ_mean'][-1]:.4f}")
    print(f"최종 목표: {history['target'][-1]:.4f} (Perf_EMA={history['perf_ema'][-1]:.4f})")
    print(f"최종 단계: {history['stage'][-1]}")
    print(f"총 Cascade: {sum(history['cascade_count'])}회")
    print(f"총 성장: {total_growth}회")
    print(f"{'='*75}")

    return agi_model, fwr_controller, history


if __name__ == "__main__":
    model, controller, history = train_resnet_with_fwr_v22(epochs=80, batch_size=64)
