import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

num_samples = 200
band1 = np.random.randn(num_samples)
band2 = 0.8 * band1 + 0.3 * np.random.randn(num_samples)

band1_centered = band1 - band1.mean()
band2_centered = band2 - band2.mean()
data_centered = np.column_stack([band1_centered, band2_centered])

covariance_matrix = np.cov(data_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

sort_order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_order]
eigenvectors = eigenvectors[:, sort_order]

principal_direction = eigenvectors[:, 0]
minor_direction = eigenvectors[:, 1]

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(band1_centered, band2_centered, alpha=0.4, s=20, color='gray', label='Samples')

scale_principal = 2 * np.sqrt(eigenvalues[0])
scale_minor = 2 * np.sqrt(eigenvalues[1])

ax.annotate('', xy=principal_direction * scale_principal, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.annotate('', xy=-principal_direction * scale_principal, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

ax.annotate('', xy=minor_direction * scale_minor, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax.annotate('', xy=-minor_direction * scale_minor, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))

ax.text(principal_direction[0] * scale_principal * 1.1,
        principal_direction[1] * scale_principal * 1.1,
        f'Eigenvector 1 (λ={eigenvalues[0]:.2f})', color='red', fontsize=10)

ax.text(minor_direction[0] * scale_minor * 1.1,
        minor_direction[1] * scale_minor * 1.1,
        f'Eigenvector 2 (λ={eigenvalues[1]:.2f})', color='blue', fontsize=10)

ax.set_xlabel('Band 1')
ax.set_ylabel('Band 2')
ax.set_title('PCA: Eigenvectors of Covariance Matrix')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print(f"Covariance matrix:\n{covariance_matrix}\n")
print(f"Eigenvalue 1 (principal): {eigenvalues[0]:.4f}")
print(f"Eigenvalue 2 (minor):     {eigenvalues[1]:.4f}")
print(f"Variance explained by eigenvector 1: {eigenvalues[0] / eigenvalues.sum() * 100:.1f}%")
