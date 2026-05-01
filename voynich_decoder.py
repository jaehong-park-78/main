import matplotlib.pyplot as plt
import numpy as np

def voynich_decoder(text):
    # 박 설계자의 해독 로직 테이블 (FWR Logic Gates)
    logic_gates = {
        'o': {'type': 'sine', 'amp': 1.0, 'freq': 1, 'desc': 'Baseline (F)'},
        'a': {'type': 'sine', 'amp': 0.8, 'freq': 1, 'desc': 'Baseline (F)'},
        'e': {'type': 'pulse', 'amp': 1.2, 'freq': 5, 'desc': 'Data Stream (W)'},
        'i': {'type': 'pulse', 'amp': 1.2, 'freq': 8, 'desc': 'Data Stream (W)'},
        't': {'type': 'spike', 'amp': 3.0, 'freq': 20, 'desc': 'Trigger (R)'},
        'p': {'type': 'spike', 'amp': 3.5, 'freq': 25, 'desc': 'Trigger (R)'},
        'k': {'type': 'spike', 'amp': 4.0, 'freq': 30, 'desc': 'Trigger (R)'},
        'm': {'type': 'mod', 'amp': 1.5, 'freq': 3, 'desc': 'Coupling (R)'},
        'n': {'type': 'mod', 'amp': 1.5, 'freq': 4, 'desc': 'Coupling (R)'},
        ' ': {'type': 'hold', 'amp': 0.0, 'freq': 0, 'desc': 'Hold-off'}
    }

    t_total = np.linspace(0, len(text), len(text) * 100)
    y_total = np.zeros_like(t_total)

    print(f"--- [FWR 인출 시작: {text}] ---")
    
    for i, char in enumerate(text.lower()):
        gate = logic_gates.get(char, {'type': 'sine', 'amp': 0.1, 'freq': 1, 'desc': 'Noise'})
        start_idx = i * 100
        end_idx = (i + 1) * 100
        t_segment = t_total[start_idx:end_idx]
        
        if gate['type'] == 'sine':
            y_total[start_idx:end_idx] = gate['amp'] * np.sin(2 * np.pi * gate['freq'] * t_segment)
        elif gate['type'] == 'pulse':
            y_total[start_idx:end_idx] = gate['amp'] * np.sign(np.sin(2 * np.pi * gate['freq'] * t_segment))
        elif gate['type'] == 'spike':
            y_total[start_idx:end_idx] = gate['amp'] * (np.abs(np.sin(2 * np.pi * gate['freq'] * t_segment))**50)
        elif gate['type'] == 'mod':
            y_total[start_idx:end_idx] = gate['amp'] * np.sin(2 * np.pi * gate['freq'] * t_segment) * np.sin(t_segment * 0.5)
        elif gate['type'] == 'hold':
            y_total[start_idx:end_idx] = 0

    # 시각화 (오실로스코프 잔상 재현)
    plt.figure(figsize=(12, 4))
    plt.plot(t_total, y_total, color='#00FF00', linewidth=1.5, label='FWR Waveform')
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    plt.fill_between(t_total, y_total, color='#00FF00', alpha=0.1)
    plt.title(f"Voynich Phase Reconstruction: '{text}'", color='white')
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.grid(True, alpha=0.1)
    plt.show()

# 실행 예시
if __name__ == "__main__":
    voynich_decoder("olteeody")
