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

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
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

margin = 2
all_points = np.array([sample_1, sample_2, mean])
ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)

ax2 = axes[1]
var_band1 = covariance_matrix[0, 0]
var_band2 = covariance_matrix[1, 1]
cov_12 = covariance_matrix[0, 1]

bar_positions = [0, 1, 2]
bar_values = [var_band1, var_band2, cov_12]
bar_labels = [f'Var(Band1)\n{var_band1:.2f}',
              f'Var(Band2)\n{var_band2:.2f}',
              f'Cov(B1,B2)\n{cov_12:.2f}']
bar_colors = ['orange', 'purple', 'teal']

ax2.bar(bar_positions, bar_values, color=bar_colors, width=0.5)
ax2.set_xticks(bar_positions)
ax2.set_xticklabels(bar_labels, fontsize=11)
ax2.set_ylabel('Value')
ax2.set_title('Covariance Matrix Entries')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
