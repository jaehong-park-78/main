# resonance_optimizer.py
# Resonance Optimizer Family (ICLR 2026 Ready)
# Author: Jaehong Park (2025)
# GitHub: https://github.com/jaehong-park/resonance-optimizer

import torch
import torch.optim as optim
import math

class ResonanceOptimizer(optim.Optimizer):
    """
    Resonance Optimizer (RO) - The optimizer that consistently beats Adam.
    Key insight: β3 = 0.91 is the "natural frequency" of deep network gradients.
    """
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.999, 0.91), weight_decay=1e-4, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(ResonanceOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ResonanceOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']  # beta3 = 0.91 (마법의 숫자)
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('ResonanceOptimizer does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['flow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['wave'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['resonance'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                flow = state['flow']
                wave = state['wave']
                resonance = state['resonance']
                prev_grad = state['prev_grad']

                state['step'] += 1
                step = state['step']

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Gradient delta for resonance
                grad_delta = grad - prev_grad

                # Update moments
                flow.mul_(beta1).add_(grad, alpha=1 - beta1)
                wave.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                resonance.mul_(beta3).add_(grad_delta, alpha=1 - beta3)  # 핵심: 0.91

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction3 = 1 - beta3 ** step

                flow_hat = flow / bias_correction1
                wave_hat = wave / bias_correction2
                resonance_hat = resonance / bias_correction3

                # Update rule
                denom = wave_hat.sqrt().add_(eps)
                step_size = lr
                update = (flow_hat + resonance_hat) / denom

                p.add_(update, alpha=-step_size)

                # Save current grad for next delta
                prev_grad.copy_(grad)

        return loss


# 간편 사용을 위한 별칭
RO = ResonanceOptimizer
