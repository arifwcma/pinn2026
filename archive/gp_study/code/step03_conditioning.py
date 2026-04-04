import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean_a, mean_b = 0.0, 0.0
var_a, var_b = 1.0, 1.0
covariance_ab = 0.8

observed_a = 2.0

conditional_mean_b = mean_b + covariance_ab / var_a * (observed_a - mean_a)
conditional_var_b = var_b - (covariance_ab ** 2) / var_a

b_values = np.linspace(-4, 4, 300)
prior_pdf = norm.pdf(b_values, mean_b, np.sqrt(var_b))
conditional_pdf = norm.pdf(b_values, conditional_mean_b, np.sqrt(conditional_var_b))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

samples_joint = np.random.multivariate_normal(
    [mean_a, mean_b],
    [[var_a, covariance_ab], [covariance_ab, var_b]],
    2000
)
axes[0].scatter(samples_joint[:, 0], samples_joint[:, 1], alpha=0.2, s=5, color="steelblue")
axes[0].axvline(observed_a, color="red", linewidth=2, linestyle="--", label=f"Observed A = {observed_a}")
axes[0].set_xlabel("Variable A")
axes[0].set_ylabel("Variable B")
axes[0].set_title("Joint Distribution + Observation")
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[0].set_aspect("equal")
axes[0].legend()

axes[1].plot(b_values, prior_pdf, color="gray", linewidth=2, label="Prior belief about B")
axes[1].plot(b_values, conditional_pdf, color="red", linewidth=2, label=f"After observing A = {observed_a}")
axes[1].axvline(conditional_mean_b, color="red", linestyle=":", alpha=0.5)
axes[1].set_xlabel("Variable B")
axes[1].set_ylabel("Probability Density")
axes[1].set_title("How Observation Updates Belief")
axes[1].legend()

print(f"Before observing A:  B ~ N(mean={mean_b}, var={var_b})")
print(f"After observing A=2: B ~ N(mean={conditional_mean_b:.2f}, var={conditional_var_b:.2f})")

plt.tight_layout()
plt.show()
