import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")
x_raw = df["size_sqm"].values
y = df["price_lakh"].values
num_samples = len(x_raw)

x_plot = np.linspace(10, 140, 200)

ones = np.ones(num_samples)
X = np.column_stack([ones, x_raw])
ols_weights = np.linalg.inv(X.T @ X) @ X.T @ y
ols_line = ols_weights[0] + ols_weights[1] * x_plot

noise_variance = 2.25
prior_mean = np.array([0.0, 0.0])
prior_covariance = np.array([[100.0, 0.0], [0.0, 100.0]])

prior_precision = np.linalg.inv(prior_covariance)
data_precision = (1.0 / noise_variance) * (X.T @ X)
blr_posterior_cov = np.linalg.inv(prior_precision + data_precision)
blr_posterior_mean = blr_posterior_cov @ (prior_precision @ prior_mean + (1.0 / noise_variance) * X.T @ y)

blr_line = blr_posterior_mean[0] + blr_posterior_mean[1] * x_plot
blr_std = np.zeros(len(x_plot))
for i, xv in enumerate(x_plot):
    x_star = np.array([1.0, xv])
    blr_std[i] = np.sqrt(x_star @ blr_posterior_cov @ x_star + noise_variance)

def rbf_kernel(x1, x2, length_scale=30.0, signal_variance=50.0):
    diff = x1[:, None] - x2[None, :]
    return signal_variance * np.exp(-0.5 * (diff / length_scale) ** 2)

gp_noise = noise_variance
K_train = rbf_kernel(x_raw, x_raw) + gp_noise * np.eye(num_samples)
K_pred_train = rbf_kernel(x_plot, x_raw)
K_pred_pred = rbf_kernel(x_plot, x_plot)

K_train_inv = np.linalg.inv(K_train)
gp_mean = K_pred_train @ K_train_inv @ y
gp_cov = K_pred_pred - K_pred_train @ K_train_inv @ K_pred_train.T
gp_std = np.sqrt(np.diag(gp_cov) + gp_noise)

gp_samples = np.random.multivariate_normal(gp_mean, gp_cov, size=5)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Same Data, Three Methods", fontsize=18, fontweight="bold", y=1.02)

ax = axes[0]
ax.scatter(x_raw, y, color="steelblue", s=60, zorder=5, edgecolor="white")
ax.plot(x_plot, ols_line, color="red", linewidth=2)
ax.set_title("OLS: One straight line\nNo uncertainty", fontsize=12, fontweight="bold")
ax.set_xlabel("House Size (sqm)")
ax.set_ylabel("Price (lakh)")
ax.set_ylim(-30, 400)

ax = axes[1]
ax.scatter(x_raw, y, color="steelblue", s=60, zorder=5, edgecolor="white")
ax.plot(x_plot, blr_line, color="red", linewidth=2, label="Mean")
ax.fill_between(x_plot, blr_line - 2 * blr_std, blr_line + 2 * blr_std,
                alpha=0.2, color="red", label="±2σ")
ax.set_title("BLR: Straight line + uncertainty\nForced to be linear", fontsize=12, fontweight="bold")
ax.set_xlabel("House Size (sqm)")
ax.set_ylabel("Price (lakh)")
ax.set_ylim(-30, 400)
ax.legend(fontsize=9)

ax = axes[2]
ax.scatter(x_raw, y, color="steelblue", s=60, zorder=5, edgecolor="white")
ax.plot(x_plot, gp_mean, color="red", linewidth=2, label="Mean")
ax.fill_between(x_plot, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std,
                alpha=0.2, color="red", label="±2σ")
for i, s in enumerate(gp_samples):
    label = "Sample curves" if i == 0 else None
    ax.plot(x_plot, s, color="red", alpha=0.15, linewidth=1, label=label)
ax.set_title("GP: Flexible curve + uncertainty\nNo shape assumed", fontsize=12, fontweight="bold")
ax.set_xlabel("House Size (sqm)")
ax.set_ylabel("Price (lakh)")
ax.set_ylim(-30, 400)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("ols_blr_gp_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to ols_blr_gp_comparison.png")
