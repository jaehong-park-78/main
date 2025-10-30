#FWR Self-Organizing Evolution Model
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ======================================================
# 1️⃣ R 계산 함수 (엔트로피 + 상관 기반)
# ======================================================
def compute_r_metric_torch(flow, wave, eps=1e-8):
    """
    R = (1 - S_out / S_in) * corr(flow, flow*wave)
    완전 미분 가능 + 안정화 버전
    """
    output = flow * wave

    # 확률 분포화
    input_prob = torch.abs(flow) / (torch.sum(torch.abs(flow)) + eps)
    output_prob = torch.abs(output) / (torch.sum(torch.abs(output)) + eps)

    # Shannon 엔트로피 (gradient-friendly)
    s_input = -torch.sum(input_prob * torch.log(input_prob + eps))
    s_output = -torch.sum(output_prob * torch.log(output_prob + eps))

    # 상관계수 (gradient-friendly)
    f_mean, o_mean = torch.mean(flow), torch.mean(output)
    cov = torch.mean((flow - f_mean) * (output - o_mean))
    std_f = torch.std(flow) + eps
    std_o = torch.std(output) + eps
    corr = cov / (std_f * std_o)

    # 효율 계산
    entropy_eff = 1 - (s_output / (s_input + eps))
    r_metric = entropy_eff * corr

    return r_metric, entropy_eff, corr, s_input, s_output


# ======================================================
# 2️⃣ FWR Self-Organizing Evolution Model
# ======================================================
def evolve_fwr_system(flow_len=256, epochs=400, lr_f=0.004, lr_w=0.008,
                      beta=0.6, gamma=0.97, device='cpu'):
    """
    FWR 자기조직화 시스템 (시간진화형)
    - Flow ↔ Wave 쌍방향 학습
    - R(t) 피드백 적용
    - gamma: 시간 감쇠율 (물리적 저항, 에너지 손실 표현)
    """
    # 초기 신호
    flow = torch.randn(flow_len, dtype=torch.float32, requires_grad=True, device=device)
    wave = torch.sin(torch.linspace(0, 2 * torch.pi, flow_len)).to(device)
    wave = (wave + 0.05 * torch.randn_like(wave)).requires_grad_()

    opt_f = torch.optim.Adam([flow], lr=lr_f)
    opt_w = torch.optim.Adam([wave], lr=lr_w)

    # 기록용
    r_hist, eff_hist, corr_hist, entropy_hist = [], [], [], []

    for epoch in range(epochs):
        opt_f.zero_grad()
        opt_w.zero_grad()

        # R 계산
        r_metric, eff, corr, s_in, s_out = compute_r_metric_torch(flow, wave)
        loss = -r_metric

        # 비대칭 업데이트 (F ↔ W)
        (beta * loss).backward(retain_graph=True)
        opt_f.step()
        loss.backward()
        opt_w.step()

        # 감쇠 적용 (물리적 시간항)
        flow.data *= gamma
        wave.data *= gamma

        # 시간 진화 기록
        r_hist.append(r_metric.item())
        eff_hist.append(eff.item())
        corr_hist.append(corr.item())
        entropy_hist.append((s_in.item(), s_out.item()))

        if epoch % 40 == 0:
            print(f"[{epoch}] R={r_metric.item():.4f} | Eff={eff.item():.3f} | Corr={corr.item():.3f}")

    return flow.detach().cpu().numpy(), wave.detach().cpu().numpy(), r_hist, eff_hist, corr_hist, entropy_hist


# ======================================================
# 3️⃣ 시각화
# ======================================================
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    flow, wave, r_hist, eff_hist, corr_hist, entropy_hist = evolve_fwr_system(
        flow_len=256, epochs=300, lr_f=0.004, lr_w=0.008, beta=0.7, gamma=0.99
    )

    s_in_hist = [s[0] for s in entropy_hist]
    s_out_hist = [s[1] for s in entropy_hist]

    plt.figure(figsize=(13, 5))

    plt.subplot(1, 3, 1)
    plt.plot(r_hist, color='blue')
    plt.title("R-Metric Temporal Evolution")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(eff_hist, label='Entropy Efficiency')
    plt.plot(corr_hist, label='Correlation')
    plt.legend()
    plt.title("Efficiency vs Correlation")

    plt.subplot(1, 3, 3)
    plt.plot(flow, label='Flow(t)')
    plt.plot(wave, label='Wave(t)')
    plt.plot(flow * wave, label='Resonance(t)')
    plt.legend()
    plt.title("Dynamic Flow-Wave Coupling")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal R: {r_hist[-1]:.4f}")
