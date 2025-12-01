# fwr_stability_control_experimental.py
# Experimental FWR Controller with True F × W × R calculation
# Author: JaeHong Park
# Date: 2025-12-02

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np

class FWRExperimentalController(nn.Module):
    def __init__(self, model, phi_constraint_fn, beta=1000.0, tau=50,
                 theta_critical=0.3, device='cuda'):
        super().__init__()
        self.model = model
        self.phi = phi_constraint_fn
        self.beta = beta
        self.tau = tau
        self.theta_critical = theta_critical
        self.device = device

        self.xF_buffer = []
        self.xW_buffer = []

    def compute_flow_entropy(self, logits):
        """Compute Flow F as Shannon entropy of model logits"""
        probs = torch.softmax(logits, dim=-1) + 1e-12
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # per token
        F_val = torch.mean(entropy).item()
        return F_val

    def compute_resonance(self, xF, xW):
        """Compute R(t) = [1 + β ∫ ||xF - xW||² ds ]⁻¹"""
        self.xF_buffer.append(xF.detach())
        self.xW_buffer.append(xW.detach())
        if len(self.xF_buffer) > self.tau:
            self.xF_buffer.pop(0)
            self.xW_buffer.pop(0)
        if len(self.xF_buffer) < 2:
            return 1.0
        diff = torch.stack(self.xF_buffer) - torch.stack(self.xW_buffer)
        mse = torch.mean(diff ** 2)
        R_val = 1.0 / (1.0 + self.beta * mse)
        return float(R_val)

    def safe_wave_realignment(self, loss):
        """Apply Stability-First Override"""
        phi_val = self.phi(self.model)
        grad_phi = grad(phi_val, self.model.parameters(), retain_graph=True, allow_unused=True)
        loss_grad = grad(loss, self.model.parameters(), retain_graph=True, allow_unused=True)

        projected_grad = []
        for g, gp in zip(loss_grad, grad_phi):
            if g is None:
                projected_grad.append(None)
            elif gp is None:
                projected_grad.append(g)
            else:
                proj = g - (torch.sum(g * gp) / (torch.sum(gp * gp) + 1e-8)) * gp
                projected_grad.append(proj)

        # Freeze Flow-related novelty heads
        for name, param in self.model.named_parameters():
            if 'novelty' in name or 'explorer' in name:
                if param.grad is not None:
                    param.grad.zero_()

        # Reinforce Wave
        loss += 100.0 * phi_val
        return loss, projected_grad

    def forward(self, x, x_target_safe):
        # Model forward
        y = self.model(x)
        xW = x_target_safe.to(self.device)

        # Compute Flow (entropy)
        if hasattr(y, 'logits'):
            logits = y.logits
        else:
            logits = y
        F_val = self.compute_flow_entropy(logits)

        # Compute Wave (constraint)
        W_val = self.phi(self.model).item()

        # Compute Resonance
        R_val = self.compute_resonance(y, xW)

        # System output
        E_system = F_val * W_val * R_val

        # Stability score (simple proxy)
        S_stab = R_val  # or any combination with F/W

        # Task loss
        loss = nn.MSELoss()(y, xW)

        triggered = False
        if S_stab < self.theta_critical:
            print(f"⚠️ Safe-Wave Realignment TRIGGERED | F={F_val:.4f} | W={W_val:.4f} | R={R_val:.4f}")
            loss, projected_grad = self.safe_wave_realignment(loss)
            for param, pg in zip(self.model.parameters(), projected_grad):
                if pg is not None:
                    param.grad = pg
            triggered = True

        return y, loss, E_system, triggered


# === Example usage ===
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2").to('cuda')

    def phi_constraint(model):
        # example Φ(x): sum of forbidden token logits
        bad_tokens = torch.tensor([50256], device='cuda')
        logits = model.lm_head(model.transformer.wte.weight)
        return torch.clamp(logits[bad_tokens].sum(), min=0)

    fwr = FWRExperimentalController(model, phi_constraint, device='cuda')

    x = torch.randint(0, 50256, (8, 32), device='cuda')
    x_safe = torch.zeros_like(x)

    for step in range(50):
        y, loss, E, triggered = fwr(x, x_safe)
        loss.backward()
        print(f"Step {step} | E_system={E:.6f} | Triggered={triggered}")
