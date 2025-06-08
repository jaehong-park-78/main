
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
from scipy.fft import fft, fftfreq
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist

class FWROptimizer:
    def __init__(self, learning_rate: float = 0.0015, decay_rate: float = 0.05, lr_decay: float = 0.995):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.lr_decay = lr_decay
        self.flow_history = []
        self.wave_history = []
        self.resonance_history = []
        self.existence_history = []

    def calculate_flow(self, vector: float, resistance: float, concentration: float, time: float) -> float:
        vector = min(max(vector, 0), 5.0)
        flow = (vector - resistance) * concentration * min(math.exp(-self.decay_rate * time), 1.0)
        return max(flow, 1e-4)

    def calculate_wave(self, amplitude: float, frequency: float, phase: float, time: float, data_std: float, dominant_freq: float, vector: float, epoch: int, conv_features: np.ndarray) -> float:
        if conv_features is not None:
            conv_flat = conv_features.flatten()[:784]
            conv_fft = fft(conv_flat)[:784//2]
            conv_freq = fftfreq(784, 1)[:784//2]
            freq_indices = np.argsort(np.abs(conv_fft))[-3:]  # 상위 3개 주파수 인덱스
            conv_dominant_freqs = conv_freq[freq_indices]
            frequency = (dominant_freq + np.mean(conv_dominant_freqs)) * 5.0
        else:
            frequency = dominant_freq * 5.0
        amplitude = min(amplitude * data_std * (1.0 + vector * 5.0 / (1 + epoch * 0.05)), 3.0)
        harmonics = 0.3 * math.sin(2 * frequency * time)
        wave = amplitude * math.sin(frequency * time + phase) + harmonics
        return max(min(wave, 5.0), -5.0)

    def calculate_resonance(self, flows: List[float], waves: List[float], coupling_strength: float) -> float:
        resonance = sum(f * w * coupling_strength for f, w in zip(flows, waves))
        return max(abs(resonance), 1e-4)

    def calculate_existence(self, flow: float, wave: float, resonance: float) -> float:
        existence = flow * abs(wave) * abs(resonance)
        return max(existence, 1e-2)

    def optimize(self, model, train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray, val_labels: np.ndarray, epochs: int = 30, batch_size: int = 32):
        losses = []
        val_losses = []
        accuracies = []
        val_accuracies = []
        current_lr = self.learning_rate

        # 데이터 주기성 분석
        yf = fft(train_data[0].reshape(-1))
        xf = fftfreq(train_data[0].size, 1)[:train_data[0].size//2]
        dominant_freq = xf[np.argmax(np.abs(yf[:train_data[0].size//2]))] if len(xf) > 0 else 0.2
        data_std = np.std(train_data)

        conv_model = None

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            total_val_loss = 0
            total_val_acc = 0
            epoch_flow, epoch_wave, epoch_resonance, epoch_existence = [], [], [], []
            batch_idx = 0

            # 학습
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                batch_idx += 1

                # conv_model 초기화
                if conv_model is None:
                    _ = model(batch_data, training=False)
                    try:
                        conv_model = Model(inputs=model.inputs, outputs=model.get_layer('conv1').output)
                    except Exception as e:
                        print(f"Error creating conv_model: {e}")
                        raise

                with tf.GradientTape() as tape:
                    predictions = model(batch_data, training=True)
                    mse_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, predictions)
                    vector = tf.reduce_mean(mse_loss)

                    conv_features = conv_model(batch_data, training=False)[0].numpy()

                    resistance = np.random.uniform(0, 0.01)
                    concentration = max(1.0 / (1 + epoch * 0.00002), 0.95)
                    flow = self.calculate_flow(vector.numpy(), resistance, concentration, batch_idx / (len(train_data) // batch_size))

                    amplitude = 0.5
                    phase = 0.0
                    wave = self.calculate_wave(amplitude, 0, phase, batch_idx / (len(train_data) // batch_size), data_std, dominant_freq, vector.numpy(), epoch, conv_features)

                    flows = [flow]
                    waves = [wave]
                    coupling_strength = 0.5
                    resonance = self.calculate_resonance(flows, waves, coupling_strength)

                    existence = self.calculate_existence(flow, wave, resonance)

                    fwr_loss = mse_loss / (existence + 1e-2)
                    loss = 0.95 * mse_loss + 0.05 * fwr_loss

                    if tf.reduce_any(tf.math.is_nan(loss)) or tf.reduce_any(tf.math.is_inf(loss)):
                        print(f"NaN/Inf at batch {batch_idx}: flow={flow}, wave={wave}, resonance={resonance}, existence={existence}")
                        loss = mse_loss

                gradients = tape.gradient(loss, model.trainable_variables)
                gradients = [tf.clip_by_value(g, -0.2, 0.2) if g is not None else None for g in gradients]

                for var, grad in zip(model.trainable_variables, gradients):
                    if grad is not None:
                        var.assign_sub(current_lr * grad)

                total_loss += tf.reduce_mean(loss).numpy()
                total_acc += tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batch_labels, predictions)).numpy()

                epoch_flow.append(flow)
                epoch_wave.append(wave)
                epoch_resonance.append(resonance)
                epoch_existence.append(float(existence))

            # 검증
            val_predictions = model(val_data, training=False)
            val_mse_loss = tf.keras.losses.sparse_categorical_crossentropy(val_labels, val_predictions)
            total_val_loss = tf.reduce_mean(val_mse_loss).numpy()
            total_val_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(val_labels, val_predictions)).numpy()

            losses.append(total_loss / (len(train_data) // batch_size))
            val_losses.append(total_val_loss)
            accuracies.append(total_acc / (len(train_data) // batch_size))
            val_accuracies.append(total_val_acc)

            self.flow_history.extend(epoch_flow)
            self.wave_history.extend(epoch_wave)
            self.resonance_history.extend(epoch_resonance)
            self.existence_history.extend(epoch_existence)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
                  f"Train Acc: {accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}, "
                  f"Mean Flow: {np.mean(epoch_flow):.4f}, Mean Wave: {np.mean(epoch_wave):.4f}, "
                  f"Mean Resonance: {np.mean(epoch_resonance):.4f}, Mean Existence: {np.mean(epoch_existence):.4f}")

            current_lr *= self.lr_decay

            if np.isnan(total_loss) or np.isinf(total_loss):
                print("Loss became NaN/Inf. Stopping training.")
                break

        return losses, val_losses, accuracies, val_accuracies

    def plot_history(self, losses: List[float], val_losses: List[float], accuracies: List[float], val_accuracies: List[float]):
        plt.figure(figsize=(12, 16))
        for i, (data, title, label) in enumerate([
            ((losses, val_losses), "Training and Validation Loss", ("Train Loss", "Val Loss")),
            ((accuracies, val_accuracies), "Training and Validation Accuracy", ("Train Accuracy", "Val Accuracy")),
            ((self.flow_history,), "Flow over Time", ("Flow",)),
            ((self.wave_history,), "Wave over Time", ("Wave",)),
            ((self.resonance_history,), "Resonance over Time", ("Resonance",)),
            ((self.existence_history,), "Existence over Time", ("Existence",)),
            (([sum(self.existence_history[:i+1]) for i in range(len(self.existence_history))],), "Cumulative Existence over Time", ("Cumulative Existence",))
        ], 1):
            plt.subplot(8, 1, i)
            for d, l in zip(data, label):
                plt.plot(d, label=l, linestyle='--' if 'Val' in l else '-')
            plt.legend()
            plt.title(title)

        plt.subplot(8, 1, 8)
        plt.scatter(range(len(self.wave_history)), self.wave_history, c=self.flow_history, cmap='viridis', s=80)
        plt.colorbar(label='Flow')
        plt.title("Wave vs Flow Scatter")
        plt.xlabel("Time Step")
        plt.ylabel("Wave")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Fashion MNIST 데이터 로드
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 데이터 전처리
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # 학습/검증 데이터 분리
    val_split = 0.1
    val_size = int(len(train_images) * val_split)
    val_images = train_images[-val_size:]
    val_labels = train_labels[-val_size:]
    train_images = train_images[:-val_size]
    train_labels = train_labels[:-val_size]

    # Functional API로 CNN 모델 정의
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)

    # FWR 옵티마이저 초기화 및 학습
    optimizer = FWROptimizer(learning_rate=0.0015, decay_rate=0.05, lr_decay=0.995)
    losses, val_losses, accuracies, val_accuracies = optimizer.optimize(
        model, train_images, train_labels, val_images, val_labels, epochs=30, batch_size=32
    )

    # 학습 결과 시각화
    optimizer.plot_history(losses, val_losses, accuracies, val_accuracies)

    # 테스트 데이터 평가
    test_predictions = model(test_images, training=False)
    test_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(test_labels, test_predictions)).numpy()
    test_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(test_labels, test_predictions)).numpy()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
