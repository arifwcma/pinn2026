import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
num_samples = 500

covariance_positive = [[1, 0.8],
                       [0.8, 1]]

covariance_zero = [[1, 0.0],
                   [0.0, 1]]

covariance_negative = [[1, -0.8],
                       [-0.8, 1]]

samples_positive = np.random.multivariate_normal(mean, covariance_positive, num_samples)
samples_zero = np.random.multivariate_normal(mean, covariance_zero, num_samples)
samples_negative = np.random.multivariate_normal(mean, covariance_negative, num_samples)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

titles = ["Positive Covariance (0.8)", "Zero Covariance", "Negative Covariance (-0.8)"]
all_samples = [samples_positive, samples_zero, samples_negative]

for ax, samples, title in zip(axes, all_samples, titles):
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=10, color="steelblue")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("Variable A")
    ax.set_ylabel("Variable B")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.show()
