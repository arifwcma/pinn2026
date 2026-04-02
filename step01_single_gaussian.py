import numpy as np
import matplotlib.pyplot as plt

mean = 5.0
std_dev = 1.5
variance = std_dev ** 2

x_values = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 300)

bell_curve = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std_dev) ** 2)

random_samples = np.random.normal(mean, std_dev, size=1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x_values, bell_curve, color="steelblue", linewidth=2)
axes[0].axvline(mean, color="red", linestyle="--", label=f"mean = {mean}")
axes[0].axvspan(mean - std_dev, mean + std_dev, alpha=0.15, color="orange", label=f"±1σ (68% of data)")
axes[0].set_title("The Bell Curve (PDF)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Probability Density")
axes[0].legend()

axes[1].hist(random_samples, bins=40, color="steelblue", edgecolor="white", density=True)
axes[1].set_title("1000 Random Samples from This Gaussian")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()
