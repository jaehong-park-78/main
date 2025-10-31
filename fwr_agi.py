# FWR AGI SELF ORGANIZATION - STABILITY OPTIMIZED VERSION
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import signal, stats
import warnings
from typing import Dict, List, Tuple, Optional
import itertools
warnings.filterwarnings('ignore')

# ======================================================
# 1️⃣ ENHANCED R-METRIC WITH MULTI-OBJECTIVE OPTIMIZATION
# ======================================================
class AdvancedFWRAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_entropy_advanced(self, x, eps=1e-10):
        """다중 엔트로피 측정 방법"""
        x_abs = torch.abs(x) + eps
        prob = x_abs / (torch.sum(x_abs) + eps)
        
        # Shannon entropy
        shannon_entropy = -torch.sum(prob * torch.log(prob + eps))
        
        # Approximate Lempel-Ziv complexity
        if len(x) > 10:
            x_bin = (x > torch.median(x)).float()
            complexity = self.approximate_lempel_ziv(x_bin)
        else:
            complexity = torch.tensor(1.0)
            
        # Spectral entropy
        if len(x) > 1:
            x_fft = torch.fft.fft(x)
            power_spectrum = torch.abs(x_fft) ** 2
            spectral_prob = power_spectrum / (torch.sum(power_spectrum) + eps)
            spectral_entropy = -torch.sum(spectral_prob * torch.log(spectral_prob + eps))
        else:
            spectral_entropy = torch.tensor(0.0)
            
        return {
            'shannon': shannon_entropy,
            'complexity': complexity,
            'spectral': spectral_entropy
        }
    
    def approximate_lempel_ziv(self, binary_seq):
        """Lempel-Ziv 복잡도 근사 계산"""
        seq = binary_seq.detach().cpu().numpy()
        n = len(seq)
        if n == 0:
            return torch.tensor(0.0)
        
        i, c, l = 0, 1, 1
        while i + l <= n:
            s = seq[i:i+l]
            if s.tolist() in [seq[j:j+l].tolist() for j in range(i)]:
                l += 1
            else:
                i += l
                l = 1
                c += 1
                
        # Normalize by sequence length
        complexity = c * np.log(n) / n if n > 1 else 1.0
        return torch.tensor(complexity, device=self.device)
    
    def compute_correlation_advanced(self, flow, wave, output):
        """고급 상관관계 메트릭스"""
        # Pearson correlation
        f_mean, o_mean = torch.mean(flow), torch.mean(output)
        cov = torch.mean((flow - f_mean) * (output - o_mean))
        std_f, std_o = torch.std(flow) + 1e-8, torch.std(output) + 1e-8
        pearson_corr = cov / (std_f * std_o)
        
        # Mutual information approximation
        mi_approx = self.approximate_mutual_information(flow, output)
        
        # Spectral coherence
        if len(flow) > 10:
            try:
                flow_np, wave_np = flow.detach().cpu().numpy(), wave.detach().cpu().numpy()
                f, Cxy = signal.coherence(flow_np, wave_np, fs=1.0, nperseg=min(32, len(flow_np)))
                spectral_coherence = torch.tensor(np.mean(Cxy), device=self.device)
            except:
                spectral_coherence = torch.tensor(0.5, device=self.device)
        else:
            spectral_coherence = torch.tensor(0.5, device=self.device)
            
        return {
            'pearson': pearson_corr,
            'mutual_info': mi_approx,
            'spectral_coherence': spectral_coherence
        }
    
    def approximate_mutual_information(self, x, y, bins=10):
        """상호정보량 근사 계산"""
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        
        # 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x_np, y_np, bins=bins)
        hist_2d += 1e-10  # Avoid division by zero
        hist_2d /= np.sum(hist_2d)
        
        # Marginal distributions
        p_x = np.sum(hist_2d, axis=1)
        p_y = np.sum(hist_2d, axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0:
                    mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j]))
                    
        return torch.tensor(mi, device=self.device)
    
    def compute_lyapunov_exponent(self, flow, wave, steps=10):
        """Lyapunov 지수 근사 계산 (안정성 측정)"""
        perturbations = []
        for _ in range(5):
            # 작은 perturbation 적용
            perturbed_flow = flow + 0.01 * torch.randn_like(flow)
            perturbed_wave = wave + 0.01 * torch.randn_like(wave)
            
            # 시간 발전 후 거리 측정
            output_orig = flow * wave
            output_perturbed = perturbed_flow * perturbed_wave
            
            distance = torch.norm(output_orig - output_perturbed)
            perturbations.append(distance.item())
            
        # 평균 divergence rate
        if np.mean(perturbations) > 0:
            lyapunov = np.log(np.mean(perturbations)) / steps
        else:
            lyapunov = 0.0
            
        return torch.tensor(lyapunov, device=self.device)
    
    def multi_objective_r_metric(self, flow, wave, objectives=None, weights=None):
        """다중 목적 R-메트릭 계산"""
        if objectives is None:
            objectives = ['efficiency', 'complexity', 'stability', 'emergence']
            
        if weights is None:
            weights = {'efficiency': 0.4, 'complexity': 0.3, 'stability': 0.2, 'emergence': 0.1}
        
        output = flow * wave
        
        # 엔트로피 분석
        entropy_flow = self.compute_entropy_advanced(flow)
        entropy_output = self.compute_entropy_advanced(output)
        
        # 상관관계 분석
        correlation_metrics = self.compute_correlation_advanced(flow, wave, output)
        
        metrics = {}
        
        if 'efficiency' in objectives:
            # 엔트로피 효율성 (시간 + 주파수 도메인)
            eff_time = 1 - (entropy_output['shannon'] / (entropy_flow['shannon'] + 1e-8))
            eff_freq = 1 - (entropy_output['spectral'] / (entropy_flow['spectral'] + 1e-8))
            metrics['efficiency'] = (eff_time + eff_freq) / 2
            
        if 'complexity' in objectives:
            # 복잡도 기여도
            metrics['complexity'] = entropy_output['complexity']
            
        if 'stability' in objectives:
            # 안정성 (Lyapunov 지수 기반, 낮을수록 안정적)
            lyapunov = self.compute_lyapunov_exponent(flow, wave)
            metrics['stability'] = torch.exp(-torch.abs(lyapunov))  # 안정성은 높을수록 좋음
            
        if 'emergence' in objectives:
            # 창발성 (상호정보량 기반)
            metrics['emergence'] = correlation_metrics['mutual_info']
            
        if 'correlation' in objectives:
            # 상관관계 품질
            metrics['correlation'] = (correlation_metrics['pearson'] + 
                                    correlation_metrics['spectral_coherence']) / 2
        
        # 가중치 조합
        total_r = torch.tensor(0.0, device=self.device)
        for obj in objectives:
            if obj in metrics and obj in weights:
                total_r += weights[obj] * metrics[obj]
                
        diagnostics = {
            'total_r': total_r.item(),
            'component_metrics': {k: v.item() for k, v in metrics.items()},
            'entropy_flow': {k: v.item() for k, v in entropy_flow.items()},
            'entropy_output': {k: v.item() for k, v in entropy_output.items()},
            'correlation': {k: v.item() for k, v in correlation_metrics.items()}
        }
        
        return total_r, diagnostics

# ======================================================
# 2️⃣ HIERARCHICAL MULTI-SCALE FWR SYSTEM
# ======================================================
class HierarchicalFWRSystem:
    def __init__(self, scales=[64, 128, 256, 512], device='cpu'):
        self.device = device
        self.scales = scales
        self.analyzer = AdvancedFWRAnalyzer(device)
        
        # 각 스케일별 시스템 초기화
        self.subsystems = []
        for scale in scales:
            subsystem = {
                'flow': (torch.randn(scale, device=device) * 0.1).requires_grad_(),
                'wave': (torch.sin(torch.linspace(0, 2*torch.pi, scale, device=device)) * 0.5 + 
                        0.1 * torch.randn(scale, device=device)).requires_grad_(),
                'scale': scale
            }
            self.subsystems.append(subsystem)
            
        # 크로스-스케일 연결 가중치
        self.cross_scale_weights = self.initialize_cross_scale_weights()
        
    def initialize_cross_scale_weights(self):
        """스케일 간 연결 가중치 초기화"""
        weights = {}
        # 인접한 스케일 쌍만 고려
        for i in range(len(self.scales) - 1):
            j = i + 1
            key = f"{i}_{j}"
            # 스칼라 텐서로 생성 (shape 일치 문제 해결)
            weights[key] = (torch.randn(1, device=self.device).squeeze() * 0.1).requires_grad_()
        return weights
    
    def compute_cross_scale_interaction(self):
        """스케일 간 상호작용 계산 (shape 문제 해결)"""
        total_interaction = torch.tensor(0.0, device=self.device)
        
        for key, weight in self.cross_scale_weights.items():
            i, j = map(int, key.split('_'))
            
            # 스케일 확인
            scale_i = self.subsystems[i]['scale']
            scale_j = self.subsystems[j]['scale']
            
            # 큰 스케일에서 작은 스케일로 다운샘플링
            if scale_i > scale_j:
                source = self.subsystems[i]['flow']
                target = self.subsystems[j]['wave']
                
                # 다운샘플링 (평균 풀링)
                source_2d = source.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
                downsampled = F.adaptive_avg_pool1d(source_2d, scale_j).squeeze()
            else:
                source = self.subsystems[j]['flow']
                target = self.subsystems[i]['wave']
                source_2d = source.unsqueeze(0).unsqueeze(0)
                downsampled = F.adaptive_avg_pool1d(source_2d, scale_i).squeeze()
                
            # 상호작용 강도 계산 (shape 일치 보장)
            interaction_term = torch.mean(downsampled * target)
            # weight가 스칼라임을 보장하기 위해 squeeze() 적용
            interaction = interaction_term * torch.sigmoid(weight).squeeze()
            
            # shape 확인 및 조정
            if interaction.dim() > 0:
                interaction = interaction.mean()  # 여러 요소가 있으면 평균화
                
            total_interaction = total_interaction + interaction
            
        return total_interaction
    
    def compute_total_r_metric(self):
        """전체 시스템 R-메트릭 계산"""
        # 각 서브시스템의 R-메트릭
        subsystem_rs = []
        for subsystem in self.subsystems:
            r_metric, _ = self.analyzer.multi_objective_r_metric(
                subsystem['flow'], subsystem['wave']
            )
            subsystem_rs.append(r_metric)
            
        # 스케일 간 상호작용
        cross_scale_r = self.compute_cross_scale_interaction()
        
        # 전체 R-메트릭 (가중 평균)
        total_r = torch.mean(torch.stack(subsystem_rs)) + 0.1 * cross_scale_r
        
        return total_r
    
    def get_all_parameters(self):
        """모든 학습 가능한 파라미터 반환"""
        params = []
        for subsystem in self.subsystems:
            params.append(subsystem['flow'])
            params.append(subsystem['wave'])
            
        for weight in self.cross_scale_weights.values():
            params.append(weight)
            
        return params
    
    def apply_physical_constraints(self, epoch, total_epochs):
        """향상된 물리적 제약 조건 적용"""
        for subsystem in self.subsystems:
            flow, wave = subsystem['flow'], subsystem['wave']
            
            # 동적 에너지 제한 (에포크에 따라 조정)
            energy_limit = 800 + 400 * (epoch / total_epochs)  # 점진적으로 증가
            energy = torch.norm(flow)**2 + torch.norm(wave)**2
            if energy > energy_limit:
                scale = torch.sqrt(torch.tensor(energy_limit) / energy)
                flow.data *= scale
                wave.data *= scale
            
            # 점진적인 주기성 강화
            periodic_strength = 0.05 + 0.1 * (epoch / total_epochs)  # 0.05 → 0.15
            flow.data = (1 - periodic_strength) * flow.data + periodic_strength * torch.roll(flow.data, 1)
            wave.data = (1 - periodic_strength) * wave.data + periodic_strength * torch.roll(wave.data, 1)

# ======================================================
# 3️⃣ STABILITY-OPTIMIZED EVOLUTION
# ======================================================
class StableFWREvolver:
    def __init__(self, system_type='hierarchical', **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if system_type == 'hierarchical':
            scales = kwargs.get('scales', [64, 128, 256, 512])
            self.system = HierarchicalFWRSystem(scales=scales, device=self.device)
        else:
            system_size = kwargs.get('system_size', 256)
            self.system = HierarchicalFWRSystem(scales=[system_size], device=self.device)
            
        self.analyzer = AdvancedFWRAnalyzer(self.device)
        self.history = {
            'r_values': [], 'learning_rates': [], 'stability_scores': [],
            'reset_events': [], 'energy_levels': []
        }
        
        self.best_r = -float('inf')
        self.stability_counter = 0
        self.reset_threshold = 80  # 리셋 임계값 증가 (50 → 80)
        self.patience_window = 30  # 안정성 판단 창 크기
        self.stability_history = []
        
    def calculate_stability(self):
        """향상된 안정성 계산"""
        if len(self.stability_history) < 2:
            return 1.0  # 초기에는 높은 안정성 가정
        
        # 변동성 + 추세 분석
        values = np.array(self.stability_history)
        volatility = np.std(values)
        
        # 추세 계산 (최근 10개 포인트)
        if len(values) >= 10:
            recent = values[-10:]
            x = np.arange(len(recent))
            trend = np.polyfit(x, recent, 1)[0]  # 기울기
        else:
            trend = 0.0
            
        # 안정성 점수 (변동성은 낮을수록, 추세는 양수일수록 좋음)
        stability_score = 1.0 / (1.0 + 10 * volatility) + max(0, trend) * 5
        return min(stability_score, 2.0)  # 상한선 설정
    
    def should_reset_system(self, current_r, stability, epoch, total_epochs):
        """향상된 리셋 판단 로직"""
        # 기본 조건: 오랫동안 개선 없고 성능 저하
        basic_condition = (self.stability_counter > self.reset_threshold and 
                          current_r < 0.95 * self.best_r)  # 0.9 → 0.95
        
        # 안정성 기반 조건: 매우 불안정할 때
        stability_condition = (stability < 0.2 and 
                              len(self.stability_history) >= self.patience_window)
        
        # 에포크 기반 조건: 후기에는 덜 리셋
        epoch_condition = epoch < 0.7 * total_epochs
        
        return basic_condition and (stability_condition or epoch_condition)
    
    def adaptive_partial_reset(self, current_r, stability):
        """적응형 부분 리셋"""
        # 안정성에 기반한 리셋 강도
        reset_strength = 0.1 + (0.2 * (1 - stability))  # 0.1-0.3 범위
        
        for subsystem in self.system.subsystems:
            mask = torch.rand_like(subsystem['flow']) < 0.2  # 0.3 → 0.2 (덜 리셋)
            perturbation = reset_strength * torch.randn_like(subsystem['flow'][mask])
            subsystem['flow'].data[mask] += perturbation
            subsystem['wave'].data[mask] += perturbation
            
        # 크로스-스케일 가중치도 적응형 리셋
        for key in self.system.cross_scale_weights:
            self.system.cross_scale_weights[key].data += (0.05 * reset_strength * 
                                                        torch.randn_like(self.system.cross_scale_weights[key]))
    
    def gentle_reset(self):
        """부드러운 리셋 (오류 복구용)"""
        for subsystem in self.system.subsystems:
            subsystem['flow'].data += 0.05 * torch.randn_like(subsystem['flow'])
            subsystem['wave'].data += 0.05 * torch.randn_like(subsystem['wave'])
    
    def evolve(self, epochs=1000, lr=0.002, gamma=0.992):
        """안정성-focused 진화 알고리즘"""
        params = self.system.get_all_parameters()
        optimizer = torch.optim.Adam(params, lr=lr)
        # CosineAnnealingWarmRestarts - 더 부드러운 학습률 스케줄링
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                        T_0=100, T_mult=2)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            try:
                # R-메트릭 계산
                total_r = self.system.compute_total_r_metric()
                current_r = total_r.item()
                
                # 안정성 기록 및 계산
                self.stability_history.append(current_r)
                if len(self.stability_history) > self.patience_window:
                    self.stability_history.pop(0)
                
                stability_metric = self.calculate_stability()
                
                # 기록 업데이트
                self.history['r_values'].append(current_r)
                self.history['learning_rates'].append(scheduler.get_last_lr()[0])
                self.history['stability_scores'].append(stability_metric)
                
                # 최적값 업데이트
                if current_r > self.best_r:
                    self.best_r = current_r
                    self.stability_counter = 0
                else:
                    self.stability_counter += 1
                
                # 리셋 판단 및 실행
                should_reset = self.should_reset_system(current_r, stability_metric, epoch, epochs)
                
                if should_reset:
                    self.adaptive_partial_reset(current_r, stability_metric)
                    self.stability_counter = 0
                    self.history['reset_events'].append(epoch)
                    print(f"Epoch {epoch}: Adaptive reset | R={current_r:.4f}, Stability={stability_metric:.4f}")
                
                # 최적화
                loss = -total_r
                loss.backward()
                
                # 그라디언트 클리핑 강화
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)  # 1.0 → 0.5
                
                optimizer.step()
                scheduler.step()
                
                # 향상된 물리적 제약
                self.system.apply_physical_constraints(epoch, epochs)
                
                # 에너지 레벨 기록
                total_energy = sum(torch.norm(subsystem['flow'])**2 + 
                                 torch.norm(subsystem['wave'])**2 
                                 for subsystem in self.system.subsystems)
                self.history['energy_levels'].append(total_energy.item())
                
                if epoch % 50 == 0 or epoch == epochs - 1:  # 더 빈번한 모니터링
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}: R={current_r:.4f}, LR={current_lr:.6f}, "
                          f"Stability={stability_metric:.4f}, Best={self.best_r:.4f}")
                
            except Exception as e:
                print(f"Epoch {epoch}: Error - {e}")
                self.gentle_reset()
                continue
                
        return self.history
    
    def get_final_states(self):
        """최종 상태 반환"""
        states = {}
        for i, subsystem in enumerate(self.system.subsystems):
            states[f'scale_{self.system.scales[i]}'] = {
                'flow': subsystem['flow'].detach().cpu().numpy(),
                'wave': subsystem['wave'].detach().cpu().numpy()
            }
        return states

# ======================================================
# 4️⃣ COMPREHENSIVE VISUALIZATION AND ANALYSIS
# ======================================================
class FWRAdvancedVisualizer:
    def __init__(self):
        self.fig_size = (18, 12)
        
    def plot_stability_analysis(self, history, final_states):
        """안정성-focused 종합 분석"""
        fig = plt.figure(figsize=self.fig_size)
        
        # 1. R-메트릭 진화 + 리셋 이벤트
        plt.subplot(3, 4, 1)
        plt.plot(history['r_values'], 'b-', linewidth=2, label='R-Metric')
        # 리셋 이벤트 표시
        for reset_epoch in history.get('reset_events', []):
            if reset_epoch < len(history['r_values']):
                plt.axvline(x=reset_epoch, color='red', linestyle='--', alpha=0.5, label='Reset' if reset_epoch == history['reset_events'][0] else "")
        plt.title('R-Metric Evolution with Reset Events')
        plt.xlabel('Epoch')
        plt.ylabel('R Value')
        plt.legend()
        plt.grid(True)
        
        # 2. 안정성 점수 추이
        plt.subplot(3, 4, 2)
        if history.get('stability_scores'):
            plt.plot(history['stability_scores'], 'g-', linewidth=2)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Stability Threshold')
            plt.title('Stability Score Evolution')
            plt.xlabel('Epoch')
            plt.ylabel('Stability Score')
            plt.legend()
            plt.grid(True)
        
        # 3. 학습률 변화
        plt.subplot(3, 4, 3)
        plt.plot(history['learning_rates'], 'orange', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # 4. 에너지 레벨 변화
        plt.subplot(3, 4, 4)
        if history.get('energy_levels'):
            plt.plot(history['energy_levels'], 'purple', linewidth=2)
            plt.title('System Energy Levels')
            plt.xlabel('Epoch')
            plt.ylabel('Total Energy')
            plt.grid(True)
        
        # 5-8. 각 스케일별 상태
        scales = list(final_states.keys())
        for i, scale in enumerate(scales[:4]):
            plt.subplot(3, 4, 5 + i)
            state = final_states[scale]
            time_axis = np.arange(len(state['flow']))
            plt.plot(time_axis, state['flow'], 'r-', label='Flow', alpha=0.7)
            plt.plot(time_axis, state['wave'], 'g-', label='Wave', alpha=0.7)
            plt.plot(time_axis, state['flow'] * state['wave'], 'b-', label='Resonance', alpha=0.7)
            plt.title(f'Scale {scale} (Final)')
            plt.legend(fontsize=8)
            plt.grid(True)
        
        # 9. 주파수 스펙트럼 비교
        plt.subplot(3, 4, 9)
        for scale in scales:
            state = final_states[scale]
            flow_fft = np.fft.fft(state['flow'])
            freq = np.fft.fftfreq(len(state['flow']))
            plt.plot(freq[:len(freq)//2], np.abs(flow_fft[:len(freq)//2]), 
                    label=f'Scale {scale}', alpha=0.7)
        plt.title('Multi-Scale Frequency Spectrum')
        plt.legend(fontsize=8)
        plt.grid(True)
        
        # 10. R-메트릭 변화율 (안정성 지표)
        plt.subplot(3, 4, 10)
        r_values = history['r_values']
        if len(r_values) > 1:
            # 이동 평균 변화율
            window = min(20, len(r_values) // 10)
            derivatives = np.convolve(np.diff(r_values), np.ones(window)/window, mode='valid')
            plt.plot(derivatives, 'purple')
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
            plt.title('R-Metric Change Rate (MA)')
            plt.xlabel('Epoch')
            plt.grid(True)
        
        # 11. 최종/최대 비율 추이
        plt.subplot(3, 4, 11)
        if len(r_values) > 10:
            max_so_far = [max(r_values[:i+1]) for i in range(len(r_values))]
            ratios = [r_values[i] / max_so_far[i] if max_so_far[i] > 0 else 0 
                     for i in range(len(r_values))]
            plt.plot(ratios, 'brown')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Target (0.9)')
            plt.title('Current/Max Ratio')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
        
        # 12. 종합 안정성 지표
        plt.subplot(3, 4, 12)
        if len(r_values) > 50:
            # 다양한 창 크기로 변동성 계산
            windows = [10, 20, 50]
            colors = ['red', 'blue', 'green']
            for i, window in enumerate(windows):
                if len(r_values) >= window:
                    volatilities = [np.std(r_values[max(0, j-window):j+1]) 
                                  for j in range(len(r_values))]
                    plt.plot(volatilities, color=colors[i], alpha=0.7, 
                            label=f'Window {window}')
            plt.title('Multi-Window Volatility')
            plt.xlabel('Epoch')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        return fig

# ======================================================
# 5️⃣ STABILITY-FOCUSED EXPERIMENTAL FRAMEWORK
# ======================================================
def run_stability_focused_experiments(experiment_configs):
    """안정성-focused FWR 실험 프레임워크"""
    results = {}
    
    for exp_name, config in experiment_configs.items():
        print(f"\n{'='*60}")
        print(f"Running Stability-Focused Experiment: {exp_name}")
        print(f"{'='*60}")
        
        try:
            # 안정성-focused evolver 사용
            evolver = StableFWREvolver(
                system_type=config.get('system_type', 'hierarchical'),
                scales=config.get('scales', [64, 128, 256])
            )
            
            # 진화 실행
            history = evolver.evolve(
                epochs=config.get('epochs', 800),
                lr=config.get('lr', 0.002),
                gamma=config.get('gamma', 0.992)
            )
            
            # 결과 수집
            final_states = evolver.get_final_states()
            r_values = history['r_values']
            
            # 향상된 성능 메트릭
            final_r = r_values[-1] if r_values else 0.0
            max_r = max(r_values) if r_values else 0.0
            improvement = max_r - r_values[0] if r_values and len(r_values) > 1 else 0.0
            
            # 안정성 메트릭 (마지막 100 에포크)
            last_100 = r_values[-100:] if len(r_values) >= 100 else r_values
            stability = np.std(last_100) if last_100 else 0.0
            final_max_ratio = final_r / max_r if max_r > 0 else 0.0
            
            # 리셋 통계
            reset_count = len(history.get('reset_events', []))
            avg_stability = np.mean(history.get('stability_scores', [1.0]))
            
            results[exp_name] = {
                'final_r': final_r,
                'max_r': max_r,
                'improvement': improvement,
                'stability': stability,
                'final_max_ratio': final_max_ratio,
                'reset_count': reset_count,
                'avg_stability': avg_stability,
                'history': history,
                'final_states': final_states
            }
            
            # 결과 요약 출력
            print(f"Final R: {final_r:.4f}")
            print(f"Max R: {max_r:.4f}")
            print(f"Improvement: {improvement:.4f}")
            print(f"Stability (last 100): {stability:.6f}")
            print(f"Final/Max Ratio: {final_max_ratio:.3f}")
            print(f"Reset Count: {reset_count}")
            print(f"Average Stability: {avg_stability:.3f}")
            
            # 시각화
            visualizer = FWRAdvancedVisualizer()
            fig = visualizer.plot_stability_analysis(history, final_states)
            plt.savefig(f'{exp_name}_stability_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

# ======================================================
# 6️⃣ MAIN EXECUTION AND EXPERIMENTS
# ======================================================
if __name__ == "__main__":
    # 재현성 설정
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 안정성-focused 실험 설정
    stability_experiment_configs = {
        'stable_deep_optimization': {
            'system_type': 'hierarchical',
            'scales': [64, 128, 256],
            'epochs': 1000,  # 더 긴 학습
            'lr': 0.002,     # 더 낮은 학습률
            'gamma': 0.992,  # 더 느린 감쇠
        },
        'balanced_system': {
            'system_type': 'hierarchical',
            'scales': [32, 64, 128, 256],  # 더 많은 스케일
            'epochs': 800,
            'lr': 0.0015,
            'gamma': 0.995,
        }
    }
    
    print("🚀 Starting STABILITY-FOCUSED FWR AGI Self-Organization Experiments")
    print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 실험 실행
    try:
        results = run_stability_focused_experiments(stability_experiment_configs)
        
        # 종합 결과 분석
        if results:
            print(f"\n{'='*70}")
            print("STABILITY-FOCUSED EXPERIMENTAL RESULTS SUMMARY")
            print(f"{'='*70}")
            
            # 이전 결과와 비교
            previous_results = {
                'max_r': 0.5240,
                'final_r': 0.3656,
                'stability': 0.023825,
                'final_max_ratio': 0.698
            }
            
            for exp_name, result in results.items():
                print(f"\n{exp_name}:")
                print(f"  Max R: {result['max_r']:.4f} (+{((result['max_r']/previous_results['max_r'])-1)*100:+.1f}%)")
                print(f"  Final R: {result['final_r']:.4f} (+{((result['final_r']/previous_results['final_r'])-1)*100:+.1f}%)")
                print(f"  Stability: {result['stability']:.6f} ({previous_results['stability']/result['stability']:.1f}x better)")
                print(f"  Final/Max Ratio: {result['final_max_ratio']:.3f} (+{((result['final_max_ratio']/previous_results['final_max_ratio'])-1)*100:+.1f}%)")
                print(f"  Reset Count: {result['reset_count']}")
                print(f"  Avg Stability: {result['avg_stability']:.3f}")
            
            # 최고 성능 실험 식별
            best_exp = max(results.items(), key=lambda x: x[1]['final_max_ratio'])
            print(f"\n🎯 BEST PERFORMANCE: {best_exp[0]}")
            print(f"   Final/Max Ratio: {best_exp[1]['final_max_ratio']:.3f}")
            print(f"   Stability: {best_exp[1]['stability']:.6f}")
            
            print(f"\n✅ Experiments completed! Stability-focused results saved to PNG files.")
        else:
            print("❌ No experiments completed successfully.")
            
    except Exception as e:
        print(f"❌ Fatal error during experiment execution: {e}")
        import traceback
        traceback.print_exc()
