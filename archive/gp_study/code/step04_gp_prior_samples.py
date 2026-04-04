import numpy as np
import matplotlib.pyplot as plt

x_points = np.linspace(0, 5, 100)

def rbf_kernel(x1, x2, length_scale=1.0):
    diff = x1[:, None] - x2[None, :]
    return np.exp(-0.5 * (diff / length_scale) ** 2)

covariance_matrix = rbf_kernel(x_points, x_points, length_scale=1.0)
mean_vector = np.zeros(len(x_points))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

num_samples = 5
samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=num_samples)

for i in range(num_samples):
    axes[0].plot(x_points, samples[i], linewidth=1.5, label=f"Sample {i+1}")
axes[0].set_title("5 Random Functions from GP Prior")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=8)

im = axes[1].imshow(covariance_matrix, extent=[0, 5, 5, 0], cmap="viridis")
axes[1].set_title("100×100 Covariance Matrix\n(built by the kernel)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("x'")
plt.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()
