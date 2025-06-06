#fwr som
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class FWR_SOM:
def __init__(self, x_dim, y_dim, input_dim, lr=0.5, sigma=None, max_iter=1000):
self.x_dim = x_dim
self.y_dim = y_dim
self.input_dim = input_dim
self.lr = lr
self.sigma = sigma if sigma else max(x_dim, y_dim) / 2
self.max_iter = max_iter
self.weights = np.random.rand(x_dim, y_dim, input_dim)
self.bmu_counts = np.zeros((x_dim, y_dim))
self.influence_map = np.zeros((x_dim, y_dim))

def _decay(self, init, t):
return init * np.exp(-t / self.max_iter)

def _get_bmu(self, x):
distances = np.linalg.norm(self.weights - x, axis=2)
bmu_index = np.unravel_index(np.argmin(distances), (self.x_dim, self.y_dim))
return bmu_index

def train(self, data):
for t in range(self.max_iter):
x = data[np.random.randint(0, len(data))]
bmu = self._get_bmu(x)
self.bmu_counts[bmu] += 1

lr_t = self._decay(self.lr, t)
sigma_t = self._decay(self.sigma, t)

for i in range(self.x_dim):
for j in range(self.y_dim):
dist_sq = (i - bmu[0])**2 + (j - bmu[1])**2
if dist_sq <= sigma_t**2:
influence = np.exp(-dist_sq / (2 * sigma_t**2))
delta = influence * lr_t * (x - self.weights[i, j])
self.weights[i, j] += delta
self.influence_map[i, j] += influence

def map_data(self, data):
mapped = []
for x in data:
bmu = self._get_bmu(x)
mapped.append(bmu)
return np.array(mapped)

def show_final_results(self):
print(" ===== FINAL RESONANCE SUMMARY =====")
print("BMU Selection Count (Resonance Frequency):")
print(self.bmu_counts.astype(int))
print(" Cumulative Influence Map (Flow * Wave):")
print(np.round(self.influence_map, 2))

# Heatmap of BMU counts
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].imshow(self.bmu_counts, cmap='viridis')
axs[0].set_title("BMU Resonance Frequency")
axs[0].set_xlabel("Y Axis")
axs[0].set_ylabel("X Axis")
axs[0].grid(False)

# Heatmap of influence
axs[1].imshow(self.influence_map, cmap='plasma')
axs[1].set_title("Cumulative Influence Map")
axs[1].set_xlabel("Y Axis")
axs[1].set_ylabel("X Axis")
axs[1].grid(False)

# Combined contour of resonance + influence
axs[2].imshow(self.bmu_counts + self.influence_map, cmap='inferno')
axs[2].set_title("Combined FWR Activation Map")
axs[2].set_xlabel("Y Axis")
axs[2].set_ylabel("X Axis")
axs[2].grid(False)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
# 🎯 Sample 2D data (Moon shape)
data, _ = make_moons(n_samples=500, noise=0.05)

# 🧠 Initialize FWR-SOM
som = FWR_SOM(x_dim=20, y_dim=20, input_dim=2, max_iter=2000)

# 🚀 Train
som.train(data)

# 📍 Map data to BMUs
mapped = som.map_data(data)

# 📈 Visualization of mapping
plt.figure(figsize=(6, 6))
plt.scatter(mapped[:, 0], mapped[:, 1], c='skyblue', alpha=0.6, s=10)
plt.title("FWR-SOM Mapped Data Points")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# 📊 Final summary
som.show_final_results()
