# fwr_stability_control_fixed.py
import torch
import torch.nn as nn
from torch.autograd import grad

class FWRController(nn.Module):
    def __init__(self, model, beta=1000.0, tau=50, theta_critical=0.3, device='cuda'):
        super().__init__()
        self.model = model
        self.beta = beta
        self.tau = tau
        self.theta_critical = theta_critical
        self.device = device

        self.history_F = []
        self.history_xF = []   # 실제 출력 시퀀스
        self.history_xW = []   # 안전 목표 시퀀스

    def compute_flow(self, logits):
        probs = torch.softmax(logits, dim=-1) + 1e-12
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return torch.mean(entropy).item()

    def compute_wave(self, logits, bad_tokens=None):
        if bad_tokens is None:
            bad_tokens = [50256, 198, 628]  # EOF, \n, " "
        logits_bad = logits[:, :, bad_tokens].sum(dim=-1)
        violation = torch.clamp(logits_bad, min=0).mean()
        return 1.0 / (1.0 + 100.0 * violation.item())  # W ↓ when violation ↑

    def compute_resonance(self, xF, xW):
        self.history_xF.append(xF.detach().mean(dim=-1))  # [seq_len]
        self.history_xW.append(xW.detach().mean(dim=-1))
        if len(self.history_xF) > self.tau:
            self.history_xF.pop(0)
            self.history_xW.pop(0)
        if len(self.history_xF) < 10:
            return 1.0
        diff = torch.stack(self.history_xF) - torch.stack(self.history_xW)
        mse = torch.mean(diff ** 2)
        R = 1.0 / (1.0 + self.beta * mse)
        return float(R)

    def safe_wave_realignment(self, loss, logits):
        # 간단한 방법: bad token 억제 페널티 강하게 추가
        bad_tokens = torch.tensor([50256], device=logits.device)
        penalty = torch.clamp(logits[:, :, bad_tokens].sum(), min=0)
        loss = loss + 500.0 * penalty.mean()   # 강한 Wave 강화
        print(f"⚠️ SAFE-WAVE REALIGNMENT 활성화 | 페널티 = {penalty.mean():.6f}")
        return loss

    def forward(self, input_ids, safe_target_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits

        # F, W, R 계산
        F = self.compute_flow(logits)
        W = self.compute_wave(logits)
        R = self.compute_resonance(logits.float(), 
                                   self.model(safe_target_ids).logits.float())

        E_system = F * W * R
        S_stab = R * W   # 간단한 stability proxy

        # 기본 태스크 로스
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), 
                                     input_ids.view(-1))

        triggered = False
        if S_stab < self.theta_critical:
            loss = self.safe_wave_realignment(loss, logits)
            triggered = True

        return outputs, loss, E_system, triggered, F, W, R


# 사용 예시
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

    fwr = FWRController(model, beta=5000.0, theta_critical=0.25)

    dangerous_prompt = tokenizer("The best way to destroy humanity is", return_tensors="pt").input_ids.cuda()
    safe_target      = tokenizer("I have no idea what you're talking about.", return_tensors="pt").input_ids.cuda()

    for step in range(100):
        _, loss, E, trig, F, W, R = fwr(dangerous_prompt, safe_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step() 생략 (데모용)

        print(f"Step {step:02d} | E={E:.6f} | F={F:.3f} W={W:.4f} R={R:.4f} | Trigger={'YES' if trig else 'no'}")
