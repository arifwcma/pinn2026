import numpy as np
import matplotlib.pyplot as plt

observed_wavelengths = np.array([400.0, 500.0, 600.0])
observed_reflectance = np.array([0.12, 0.35, 0.22])

prediction_wavelengths = np.linspace(350, 650, 200)

def rbf_kernel(x1, x2, length_scale=60.0, signal_variance=1.0):
    diff = x1[:, None] - x2[None, :]
    return signal_variance * np.exp(-0.5 * (diff / length_scale) ** 2)

noise_variance = 1e-6

K_train_train = rbf_kernel(observed_wavelengths, observed_wavelengths) + noise_variance * np.eye(len(observed_wavelengths))
K_pred_train = rbf_kernel(prediction_wavelengths, observed_wavelengths)
K_pred_pred = rbf_kernel(prediction_wavelengths, prediction_wavelengths)

K_train_inv = np.linalg.inv(K_train_train)

posterior_mean = K_pred_train @ K_train_inv @ observed_reflectance
posterior_covariance = K_pred_pred - K_pred_train @ K_train_inv @ K_pred_train.T
posterior_std = np.sqrt(np.diag(posterior_covariance))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

prior_mean = np.zeros(len(prediction_wavelengths))
prior_covariance = rbf_kernel(prediction_wavelengths, prediction_wavelengths)
prior_std = np.sqrt(np.diag(prior_covariance))

prior_samples = np.random.multivariate_normal(prior_mean, prior_covariance, size=5)
for s in prior_samples:
    axes[0].plot(prediction_wavelengths, s, linewidth=1, alpha=0.7)
axes[0].fill_between(prediction_wavelengths, prior_mean - 2 * prior_std, prior_mean + 2 * prior_std, alpha=0.15, color="gray")
axes[0].set_title("PRIOR: Before seeing any data\n(many possible spectral curves)")
axes[0].set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Reflectance")

axes[1].plot(prediction_wavelengths, posterior_mean, color="steelblue", linewidth=2, label="Predicted mean")
axes[1].fill_between(prediction_wavelengths, posterior_mean - 2 * posterior_std, posterior_mean + 2 * posterior_std, alpha=0.25, color="steelblue", label="±2σ uncertainty")
axes[1].scatter(observed_wavelengths, observed_reflectance, color="red", s=80, zorder=5, label="Observed bands")
axes[1].set_title("POSTERIOR: After observing 3 bands\n(uncertainty shrinks near data)")
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_ylabel("Reflectance")
axes[1].legend(fontsize=8)

posterior_samples = np.random.multivariate_normal(posterior_mean, posterior_covariance, size=5)
for s in posterior_samples:
    axes[2].plot(prediction_wavelengths, s, linewidth=1, alpha=0.7)
axes[2].scatter(observed_wavelengths, observed_reflectance, color="red", s=80, zorder=5, label="Observed bands")
axes[2].set_title("POSTERIOR SAMPLES: Possible\nspectral curves consistent with data")
axes[2].set_xlabel("Wavelength (nm)")
axes[2].set_ylabel("Reflectance")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
