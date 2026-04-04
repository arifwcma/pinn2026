import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")
x_raw = df["size_sqm"].values
y = df["price_lakh"].values

noise_variance = 2.25
prior_mean = np.array([0.0, 0.0])
prior_covariance = np.array([[100.0, 0.0],
                              [0.0, 100.0]])

stages = [0, 1, 3, 10]
x_plot = np.linspace(10, 140, 200)
num_line_samples = 30

fig, axes = plt.subplots(1, 42, figsize=(22, 5))

prior_precision = np.linalg.inv(prior_covariance)

print(prior_precision)

posterior_mean = prior_mean.copy()
posterior_cov = prior_covariance.copy()

weight_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, size=num_line_samples)

print(weight_samples)

exit(0)

for panel_idx, num_observed in enumerate(stages):
    ax = axes[panel_idx]

    if num_observed == 0:
        posterior_mean = prior_mean.copy()
        posterior_cov = prior_covariance.copy()
    else:
        x_subset = x_raw[:num_observed]
        y_subset = y[:num_observed]
        ones = np.ones(num_observed)
        X_subset = np.column_stack([ones, x_subset])

        prior_precision = np.linalg.inv(prior_covariance)
        data_precision = (1.0 / noise_variance) * (X_subset.T @ X_subset)
        posterior_cov = np.linalg.inv(prior_precision + data_precision)
        posterior_mean = posterior_cov @ (prior_precision @ prior_mean + (1.0 / noise_variance) * X_subset.T @ y_subset)

    weight_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, size=num_line_samples)

    for sample_w in weight_samples:
        line = sample_w[0] + sample_w[1] * x_plot
        ax.plot(x_plot, line, color="red", alpha=0.12, linewidth=1)

    mean_line = posterior_mean[0] + posterior_mean[1] * x_plot
    ax.plot(x_plot, mean_line, color="darkred", linewidth=2)

    if num_observed > 0:
        ax.scatter(x_raw[:num_observed], y[:num_observed], color="steelblue", s=60, zorder=5, edgecolor="white")

    ax.set_xlim(10, 140)
    ax.set_ylim(-50, 400)
    ax.set_xlabel("House Size (sqm)")
    if panel_idx == 0:
        ax.set_ylabel("Price (lakh)")

    if num_observed == 0:
        ax.set_title("0 samples seen\n(prior: any line possible)", fontsize=11, fontweight="bold")
    elif num_observed == 1:
        ax.set_title("1 sample seen\n(many lines still fit)", fontsize=11, fontweight="bold")
    elif num_observed == 3:
        ax.set_title("3 samples seen\n(lines narrowing down)", fontsize=11, fontweight="bold")
    else:
        ax.set_title("All 10 samples seen\n(tight agreement)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("blr_sequential_update.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved to blr_sequential_update.png")
