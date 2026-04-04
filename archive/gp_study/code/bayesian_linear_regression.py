import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")
x_raw = df["size_sqm"].values
y = df["price_lakh"].values
num_samples = len(x_raw)

ones_column = np.ones(num_samples)
X = np.column_stack([ones_column, x_raw])

noise_variance = 2.25
prior_mean = np.array([0.0, 0.0])
prior_covariance = np.array([[100.0, 0.0],
                              [0.0, 100.0]])

prior_precision = np.linalg.inv(prior_covariance)
data_precision = (1.0 / noise_variance) * (X.T @ X)
posterior_covariance = np.linalg.inv(prior_precision + data_precision)
posterior_mean = posterior_covariance @ (prior_precision @ prior_mean + (1.0 / noise_variance) * X.T @ y)

ols_weights = np.linalg.inv(X.T @ X) @ X.T @ y

x_plot = np.linspace(10, 140, 200)

ols_prediction = ols_weights[0] + ols_weights[1] * x_plot
blr_prediction = posterior_mean[0] + posterior_mean[1] * x_plot

blr_uncertainty = np.zeros(len(x_plot))
for i, x_val in enumerate(x_plot):
    x_star = np.array([1.0, x_val])
    predictive_variance = x_star @ posterior_covariance @ x_star + noise_variance
    blr_uncertainty[i] = np.sqrt(predictive_variance)

num_line_samples = 20
weight_samples = np.random.multivariate_normal(posterior_mean, posterior_covariance, size=num_line_samples)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(x_raw, y, color="steelblue", s=60, zorder=5)
axes[0].plot(x_plot, ols_prediction, color="red", linewidth=2)
axes[0].set_title("OLS: One Best Line\n(no uncertainty)")
axes[0].set_xlabel("House Size (sqm)")
axes[0].set_ylabel("Price (lakh)")
axes[0].set_ylim(-50, 400)

axes[1].scatter(x_raw, y, color="steelblue", s=60, zorder=5)
axes[1].plot(x_plot, blr_prediction, color="red", linewidth=2, label="Posterior mean")
axes[1].fill_between(x_plot,
                      blr_prediction - 2 * blr_uncertainty,
                      blr_prediction + 2 * blr_uncertainty,
                      alpha=0.2, color="red", label="±2σ uncertainty")
axes[1].set_title("BLR: Best Line + Uncertainty\n(wider far from data)")
axes[1].set_xlabel("House Size (sqm)")
axes[1].set_ylabel("Price (lakh)")
axes[1].set_ylim(-50, 400)
axes[1].legend(fontsize=8)

axes[2].scatter(x_raw, y, color="steelblue", s=60, zorder=5)
for sample_weights in weight_samples:
    sample_line = sample_weights[0] + sample_weights[1] * x_plot
    axes[2].plot(x_plot, sample_line, color="red", alpha=0.15, linewidth=1)
axes[2].plot(x_plot, blr_prediction, color="darkred", linewidth=2, label="Posterior mean")
axes[2].set_title(f"BLR: {num_line_samples} Sampled Lines\n(all plausible given the data)")
axes[2].set_xlabel("House Size (sqm)")
axes[2].set_ylabel("Price (lakh)")
axes[2].set_ylim(-50, 400)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

print("=== OLS ===")
print(f"  Intercept: {ols_weights[0]:.4f}")
print(f"  Slope:     {ols_weights[1]:.4f}")
print()
print("=== Bayesian LR ===")
print(f"  Posterior mean:  w0={posterior_mean[0]:.4f}, w1={posterior_mean[1]:.4f}")
print(f"  Posterior covariance:")
print(f"    [{posterior_covariance[0,0]:.6f}, {posterior_covariance[0,1]:.6f}]")
print(f"    [{posterior_covariance[1,0]:.6f}, {posterior_covariance[1,1]:.6f}]")
