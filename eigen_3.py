import numpy as np
import matplotlib.pyplot as plt

sample_1 = np.array([1, 2])
sample_2 = np.array([3, 5])

data = np.array([sample_1, sample_2])
mean = data.mean(axis=0)
centered = data - mean

covariance_matrix = np.cov(centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

sort_order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_order]
eigenvectors = eigenvectors[:, sort_order]

print(f"Sample 1: {sample_1}")
print(f"Sample 2: {sample_2}")
print(f"Mean:     {mean}")
print(f"Centered sample 1: {centered[0]}")
print(f"Centered sample 2: {centered[1]}")
print(f"\nCovariance matrix:\n{covariance_matrix}")
print(f"  C[0,0] = var(band1) = {covariance_matrix[0,0]:.4f}")
print(f"  C[1,1] = var(band2) = {covariance_matrix[1,1]:.4f}")
print(f"  C[0,1] = cov(band1,band2) = {covariance_matrix[0,1]:.4f}")
print(f"\nEigenvalue 1: {eigenvalues[0]:.4f}")
print(f"Eigenvector 1: {eigenvectors[:, 0]}")
print(f"\nEigenvalue 2: {eigenvalues[1]:.4f}")
print(f"Eigenvector 2: {eigenvectors[:, 1]}")

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(data[:, 0], data[:, 1], s=100, color='black', zorder=5)
ax.annotate(f'Sample 1 ({sample_1[0]},{sample_1[1]})', xy=sample_1,
            xytext=(sample_1[0]-1.2, sample_1[1]+0.3), fontsize=10)
ax.annotate(f'Sample 2 ({sample_2[0]},{sample_2[1]})', xy=sample_2,
            xytext=(sample_2[0]+0.2, sample_2[1]+0.3), fontsize=10)

ax.scatter(*mean, s=100, color='green', marker='x', zorder=5, linewidths=2)
ax.annotate(f'Mean ({mean[0]},{mean[1]})', xy=mean,
            xytext=(mean[0]+0.2, mean[1]-0.5), fontsize=10, color='green')

principal = eigenvectors[:, 0]
minor = eigenvectors[:, 1]

ax.arrow(mean[0], mean[1], principal[0] * 2, principal[1] * 2,
         head_width=0.15, head_length=0.1, fc='red', ec='red', lw=3, zorder=10)
ax.text(mean[0] + principal[0] * 2.3, mean[1] + principal[1] * 2.3,
        f'Eigenvec 1 (λ={eigenvalues[0]:.2f})', color='red', fontsize=10, fontweight='bold')

ax.arrow(mean[0], mean[1], minor[0] * 2, minor[1] * 2,
         head_width=0.15, head_length=0.1, fc='blue', ec='blue', lw=3, zorder=10)
ax.text(mean[0] + minor[0] * 2.3, mean[1] + minor[1] * 2.3,
        f'Eigenvec 2 (λ={eigenvalues[1]:.2f})', color='blue', fontsize=10, fontweight='bold')

ax.set_xlabel('Band 1')
ax.set_ylabel('Band 2')
ax.set_title('Samples + Eigenvectors')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

var_band1 = covariance_matrix[0, 0]
var_band2 = covariance_matrix[1, 1]
cov_12 = covariance_matrix[0, 1]

cov_points = {
    f'Var(B1) = ({var_band1:.1f}, 0)': (var_band1, 0),
    f'Var(B2) = (0, {var_band2:.1f})': (0, var_band2),
    f'Cov(B1,B2) = ({cov_12:.1f}, {cov_12:.1f})': (cov_12, cov_12),
}
cov_colors = ['orange', 'purple', 'teal']
for (label, point), color in zip(cov_points.items(), cov_colors):
    ax.scatter(*point, s=150, color=color, marker='D', zorder=6, edgecolors='black')
    ax.annotate(label, xy=point, xytext=(point[0]+0.2, point[1]-0.4),
                fontsize=9, color=color, fontweight='bold')

margin = 1
all_coords = np.array([sample_1, sample_2, mean,
                        [var_band1, 0], [0, var_band2], [cov_12, cov_12]])
ax.set_xlim(-0.5, all_coords[:, 0].max() + margin)
ax.set_ylim(-0.5, all_coords[:, 1].max() + margin)

plt.tight_layout()
plt.show()
