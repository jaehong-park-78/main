import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import matplotlib.pyplot as plt
import numpy as np

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================
# 🔹 FWR Ontology 기반 Reward Function (최종 개선 버전)
# ===========================================
class FWR_R_Aligner(nn.Module):
    def __init__(self, decay_rate=0.03, initial_wave_temp=1.0, r_weight=0.3):
        super().__init__()
        self.decay_rate = decay_rate
        self.initial_wave_temp = initial_wave_temp
        self.r_weight = r_weight
        # CrossEntropyLoss는 나중에 Base Reward 계산에 사용됩니다.
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none') 

    def forward(self, response_logits, human_feedback, epoch=0):
        """
        FWR 기반 RLHF 보상함수 계산. 배치 처리 및 GPU 지원.
        E = F × |W| × |R|
        """
        B, S, V = response_logits.shape  # Batch, Sequence, Vocab

        # 1. 🌊 Wave Temperature (동적 조정)
        # 에포크가 진행될수록 온도를 낮춰 Resonance 및 Wave 구조를 안정화/정제합니다.
        current_wave_temp = self.initial_wave_temp * math.exp(-0.01 * epoch)

        # 2. F — Flow (Feedback 흐름, 배치 평균)
        F_decay = torch.exp(torch.tensor(-self.decay_rate * epoch, device=DEVICE))
        # human_feedback: [B] -> [B, 1]로 확장하여 F 계산
        F = human_feedback.unsqueeze(1) * F_decay.float() 

        # 3. W — Wave (로짓 FFT 스펙트럼 에너지, 배치 평균)
        # 로짓의 평균 스펙트럼 에너지를 통해 파형 복잡도를 측정합니다.
        logits_flat = response_logits.view(B * S, V)
        mean_logit = logits_flat.mean(dim=0, keepdim=True)  # [1, V]
        
        fft_logits = torch.fft.fft(mean_logit.float())
        spectral_energy = torch.sum(torch.abs(fft_logits) ** 2)
        
        # log-scale 안정화 후 W 계산. 배치 크기만큼 반복하여 [B, 1] 생성
        W = torch.tanh(torch.log1p(spectral_energy)).clamp(0, 1).unsqueeze(0).repeat(B, 1) 

        # 4. R — Resonance (엔트로피 기반 효율성)
        probs = torch.softmax(response_logits / current_wave_temp, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        # 시퀀스 차원 평균 엔트로피 계산 [B, 1]
        entropy = -(probs * log_probs).sum(dim=-1).mean(dim=1, keepdim=True) 
        
        # 정규화를 위한 최대 엔트로피
        max_entropy = torch.log(torch.tensor(V, device=DEVICE).float())
        
        # Resonance 효율성: 1 - (실제 엔트로피 / 최대 엔트로피)
        R = (1 - (entropy / max_entropy)).clamp(0, 1)  # [B, 1]

        # 5. E — 존재 효율 (배치별 곱)
        E = F * W * R  # [B, 1]

        # 6. Base reward (언어 일관성/정확도)
        # 다음 토큰 예측 일관성을 CE Loss로 측정합니다.
        shift_logits = response_logits[..., :-1, :].contiguous()
        shift_labels = response_logits[..., 1:, :].contiguous()
        
        # 예측된 다음 토큰을 타겟으로 사용 (Self-Supervised consistency)
        target_tokens = torch.argmax(shift_labels, dim=-1) # [B, S-1]

        # CE Loss 계산 및 시퀀스 평균 [B, 1]
        ce_loss = self.cross_entropy(
            shift_logits.view(-1, V), 
            target_tokens.view(-1)
        ).view(B, -1).mean(dim=1, keepdim=True)
        
        # 베이스 보상: 1 - CE_Loss (낮은 손실은 높은 보상)
        base_reward = (1 - ce_loss.detach()).clamp(0, 1)  # detach()를 통해 Loss는 Backprop하지 않음

        # 7. FWR 결합 최종 보상 (배치별)
        final_reward = (1 - self.r_weight) * base_reward + self.r_weight * E  # [B, 1]

        # FWR 요소 평균 메트릭 (로깅용)
        metrics = {
            'F': F.mean().item(),
            'W': W.mean().item(),
            'R': R.mean().item(),
            'E': E.mean().item(),
            'Base_Reward': base_reward.mean().item(),
            'Wave_Temp': current_wave_temp
        }

        return final_reward.squeeze(1), metrics  # [B] reward 반환

# ===========================================
# 🔹 Simple Critic for PPO Advantage
# ===========================================
# LLM의 출력을 기반으로 현재 상태의 가치(Value)를 예측하는 Critic 네트워크
class SimpleCritic(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, hidden_states):
        # Hidden State를 시퀀스 차원으로 평균하여 Context Vector 생성 후 Value 예측
        return self.fc(hidden_states.mean(dim=1))  # [B, 1]

# ===========================================
# 🔹 GPT2 + FWR-RLHF 실험 (개선됨: 배치 + PPO-like + 플롯)
# ===========================================

# 모델 및 토크나이저 초기화
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token

def generate_response(model, inputs, max_len=60, temperature=0.8):
    """모델 응답 생성 함수 (입력 텐서 받음)"""
    outputs = model.generate(
        **inputs, 
        max_length=max_len, 
        do_sample=True, 
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    return outputs

# 하이퍼파라미터
EPOCHS = 20 # 더 긴 훈련을 위해 증가
LR = 1e-5
REWARD_WEIGHT = 0.4
BATCH_SIZE = 3 
CLIP_EPSILON = 0.2 
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_BETA = 0.01

# 데이터
prompts = [
    "The essence of the FWR model is",
    "How does the mind achieve self-organization?",
    "Write a short philosophical quote about existence and flow."
]
human_feedback = torch.tensor([0.95, 0.80, 0.75], dtype=torch.float32, device=DEVICE)  # [B]

# 옵티마이저 및 보상 함수
optimizer = optim.Adam(model.parameters(), lr=LR)
reward_fn = FWR_R_Aligner(r_weight=REWARD_WEIGHT).to(DEVICE)

# Critic 초기화
critic = SimpleCritic().to(DEVICE)
opt_critic = optim.Adam(critic.parameters(), lr=LR)

# 메트릭 히스토리
history = {'rewards': [], 'F': [], 'W': [], 'R': [], 'E': [], 'base': []}

print(f"--- FWR-RLHF PPO Integration Training Started on {DEVICE} ---")
print(f"Epochs: {EPOCHS}, Learning Rate: {LR}, FWR Weight: {REWARD_WEIGHT}")

for epoch in range(EPOCHS):
    
    # --- 데이터 준비 ---
    batch_prompts = (prompts * (BATCH_SIZE // len(prompts) + 1))[:BATCH_SIZE]
    batch_feedback = human_feedback.repeat(BATCH_SIZE // len(human_feedback) + 1)[:BATCH_SIZE]
    
    # 토크나이징 (배치)
    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=50).to(DEVICE)
    
    # 모델의 동적 Wave Temperature 설정
    current_temp = reward_fn.initial_wave_temp * math.exp(-0.01 * epoch)

    # 1. 응답 생성 및 로짓 계산
    model.eval()
    with torch.no_grad():
        # Policy Network (GPT-2)로부터 응답 생성
        outputs = generate_response(model, inputs, max_len=inputs.input_ids.size(1) + 30, temperature=current_temp)
    model.train()

    # 생성된 응답에 대한 로짓 계산 (Grad tracking 활성화)
    # Note: 여기서는 생성된 outputs을 다시 입력으로 사용하여 전체 시퀀스 로짓을 계산합니다.
    output_text = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    full_inputs = tokenizer(output_text, return_tensors='pt', padding=True, truncation=True, max_length=80).to(DEVICE)
    
    logits = model(**full_inputs).logits  # [B, S, V]
    hidden_states = model(**full_inputs).last_hidden_state  # [B, S, D]

    # 2. FWR 보상 및 Value 계산
    with torch.no_grad():
        final_reward, metrics = reward_fn(logits, batch_feedback, epoch)  # [B]
        values_old = critic(hidden_states) # [B, 1]

    # --- PPO Optimization Loop ---
    # 실제 PPO에서는 여러 Mini-batch를 사용하지만, 여기서는 1번의 업데이트만 수행합니다.

    # 3. Value 예측 (Critic)
    values = critic(hidden_states)  # [B, 1]
    
    # Advantage 계산
    advantages = final_reward.unsqueeze(1) - values_old # [B, 1]
    # GAE(Generalized Advantage Estimation) 대신 Simple Advantage 사용
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 정규화

    # 4. Policy Loss (Surrogate Loss)
    # 현재 모델의 로그 확률 (Policy)
    current_log_probs = torch.log_softmax(logits, dim=-1)
    
    # 액션(선택된 토큰)에 대한 로그 확률만 추출
    action_tokens = torch.argmax(logits, dim=-1)
    current_log_probs = current_log_probs.gather(2, action_tokens.unsqueeze(-1)).squeeze(-1) # [B, S]
    
    # 이전 Policy (여기서는 단순 detach로 대체)의 로그 확률 (PPO-KLDivergence 항을 단순화함)
    # RLHF 에서는 Reference Model이 필요하지만, 여기서는 간단화를 위해 Current Policy를 사용합니다.
    ratio = torch.exp(current_log_probs.sum(dim=1, keepdim=True) - current_log_probs.sum(dim=1, keepdim=True).detach()) # R_t(theta) / R_t(theta_old)
    
    # Clipped Surrogate Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
    policy_loss = -torch.min(surr1, surr2).mean() # Policy Gradient

    # Entropy Bonus (탐색 증진)
    entropy_term = -ENTROPY_BETA * (-(current_log_probs * torch.exp(current_log_probs)).sum(dim=-1).mean())
    
    # 5. Critic Loss (Value Function Loss)
    critic_loss = CRITIC_LOSS_WEIGHT * nn.MSELoss()(values, final_reward.unsqueeze(1))

    # 6. 총 손실 및 업데이트
    total_loss = policy_loss + critic_loss + entropy_term
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

    optimizer.step(); opt_critic.step()
    optimizer.zero_grad(); opt_critic.zero_grad()

    # 7. 로깅 누적
    avg_r = final_reward.mean().item()
    
    # 메트릭 히스토리 업데이트
    history['rewards'].append(avg_r)
    for key in ['F', 'W', 'R', 'E', 'Base_Reward']:
        history[key].append(metrics[key])

    # 8. 에포크 결과 출력
    print(f"\n==================== Epoch {epoch+1}/{EPOCHS} ====================")
    print(f"🔹 Avg Final Reward: {avg_r:.4f}")
    print(f"  > FWR Coherence (E): {metrics['E']:.4f} (F: {metrics['F']:.4f} | W: {metrics['W']:.4f} | R: {metrics['R']:.4f})")
    print(f"  > Avg Base Reward: {metrics['Base_Reward']:.4f}")
    print(f"  > PPO Loss: {policy_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy Bonus: {entropy_term.item():.4f}")
    print(f"  > Dynamic Wave Temp: {current_temp:.4f}")

    # 샘플 응답 (배치 첫 번째)
    test_prompt = "FWR Ontology is the principle behind"
    test_inputs = tokenizer(test_prompt, return_tensors='pt', padding=True, truncation=True, max_length=50).to(DEVICE)
    sample_outputs = generate_response(model, test_inputs, max_len=test_inputs.input_ids.size(1) + 30, temperature=current_temp)
    sample_response = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    print(f"  > Test Response (Prompt: '{test_prompt}'):\n    {sample_response}")


# 9. 플롯 로깅 (F/W/R/E 곡선)
plt.figure(figsize=(12, 8))
plt.suptitle('FWR-RLHF PPO Integration Metrics Evolution', fontsize=16)

plt.subplot(2, 2, 1)
plt.plot(history['rewards'], label='Avg Reward', color='blue')
plt.title('Avg Final Reward')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(history['F'], label='F (Flow)', color='red')
plt.plot(history['W'], label='W (Wave)', color='green')
plt.plot(history['R'], label='R (Resonance)', color='purple')
plt.title('FWR Component Metrics (F/W/R)')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(history['E'], label='E (Coherence)', color='darkorange')
plt.title('Existence Coherence (E)')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(history['base'], label='Base Reward (Consistency)', color='gray')
plt.title('Base Reward Evolution')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('fwr_rlhf_metrics.png')
# plt.show() # Canvas 환경에서는 주석 처리

print("\n--- FWR-RLHF PPO Integration Training Finished ---")
