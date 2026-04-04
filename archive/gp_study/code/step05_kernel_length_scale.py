import numpy as np
import matplotlib.pyplot as plt

x_points = np.linspace(0, 5, 100)

def rbf_kernel(x1, x2, length_scale=1.0):
    diff = x1[:, None] - x2[None, :]
    return np.exp(-0.5 * (diff / length_scale) ** 2)

length_scales = [0.2, 1.0, 3.0]
num_samples = 4

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

for col, length_scale in enumerate(length_scales):
    covariance_matrix = rbf_kernel(x_points, x_points, length_scale)
    mean_vector = np.zeros(len(x_points))

    axes[0, col].imshow(covariance_matrix, extent=[0, 5, 5, 0], cmap="viridis")
    axes[0, col].set_title(f"Covariance Matrix\nℓ = {length_scale}")
    axes[0, col].set_xlabel("x")
    axes[0, col].set_ylabel("x'")

    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=num_samples)
    for i in range(num_samples):
        axes[1, col].plot(x_points, samples[i], linewidth=1.2)
    axes[1, col].set_title(f"Sample Functions (ℓ = {length_scale})")
    axes[1, col].set_xlabel("Location along road (km)")
    axes[1, col].set_ylabel("Elevation f(x)")
    axes[1, col].set_ylim(-3.5, 3.5)

plt.tight_layout()
plt.show()
