import numpy as np
import matplotlib.pyplot as plt

sample_1 = np.array([2, 4])
sample_2 = np.array([4, 8])

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
print(f"Centered: {centered}")
print(f"\nCovariance matrix:\n{covariance_matrix}")
print(f"\nEigenvalue 1: {eigenvalues[0]:.4f}")
print(f"Eigenvector 1: {eigenvectors[:, 0]}")
print(f"\nEigenvalue 2: {eigenvalues[1]:.4f}")
print(f"Eigenvector 2: {eigenvectors[:, 1]}")

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(data[:, 0], data[:, 1], s=100, color='black', zorder=5)
ax.annotate('Sample 1 (2,4)', xy=sample_1, xytext=(sample_1[0]-1.5, sample_1[1]+0.3), fontsize=10)
ax.annotate('Sample 2 (4,8)', xy=sample_2, xytext=(sample_2[0]+0.2, sample_2[1]+0.3), fontsize=10)

ax.scatter(*mean, s=100, color='green', marker='x', zorder=5, linewidths=2)
ax.annotate(f'Mean ({mean[0]},{mean[1]})', xy=mean, xytext=(mean[0]+0.2, mean[1]-0.5), fontsize=10, color='green')

principal = eigenvectors[:, 0]
minor = eigenvectors[:, 1]

ax.arrow(mean[0], mean[1], principal[0] * 3, principal[1] * 3,
         head_width=0.2, head_length=0.15, fc='red', ec='red', lw=3, zorder=10)
ax.text(mean[0] + principal[0] * 3.3, mean[1] + principal[1] * 3.3,
        f'Eigenvec 1 (λ={eigenvalues[0]:.2f})', color='red', fontsize=11, fontweight='bold')

ax.arrow(mean[0], mean[1], minor[0] * 3, minor[1] * 3,
         head_width=0.2, head_length=0.15, fc='blue', ec='blue', lw=3, zorder=10)
ax.text(mean[0] + minor[0] * 3.3, mean[1] + minor[1] * 3.3,
        f'Eigenvec 2 (λ={eigenvalues[1]:.2f})', color='blue', fontsize=11, fontweight='bold')

ax.set_xlabel('Band 1')
ax.set_ylabel('Band 2')
ax.set_title('Two samples, two bands')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

margin = 2
all_points = np.array([sample_1, sample_2, mean])
ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)

plt.tight_layout()
plt.show()
