import numpy as np
import matplotlib.pyplot as plt

a = np.array([2, 3])
b = np.array([4, 1])

c = np.dot(a, b) / np.dot(b, b)
projection = c * b
error = a - projection

fig, ax = plt.subplots(figsize=(8, 8))

ax.arrow(0, 0, a[0], a[1], head_width=0.1, head_length=0.08,
         fc='black', ec='black', lw=2, zorder=5)
ax.text(a[0] + 0.15, a[1] + 0.1, f'a = ({a[0]}, {a[1]})',
        fontsize=12, fontweight='bold')

ax.arrow(0, 0, b[0], b[1], head_width=0.1, head_length=0.08,
         fc='green', ec='green', lw=2, zorder=5)
ax.text(b[0] + 0.15, b[1] + 0.1, f'b = ({b[0]}, {b[1]})',
        fontsize=12, color='green', fontweight='bold')

ax.arrow(0, 0, projection[0], projection[1], head_width=0.1, head_length=0.08,
         fc='red', ec='red', lw=2.5, zorder=6)
ax.text(projection[0] + 0.15, projection[1] - 0.3,
        f'p = c·b = {c:.2f}·b = ({projection[0]:.2f}, {projection[1]:.2f})',
        fontsize=10, color='red', fontweight='bold')

ax.annotate('', xy=a, xytext=projection,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, linestyle='--'))
ax.text((a[0] + projection[0]) / 2 + 0.15, (a[1] + projection[1]) / 2,
        f'e = a - p = ({error[0]:.2f}, {error[1]:.2f})',
        fontsize=10, color='blue', fontweight='bold')

perp_size = 0.2
perp_dir = error / np.linalg.norm(error)
along_dir = b / np.linalg.norm(b)
corner = projection + perp_dir * perp_size
corner2 = corner + along_dir * perp_size
corner3 = projection + along_dir * perp_size
ax.plot([corner[0], corner2[0]], [corner[1], corner2[1]], 'k-', lw=1)
ax.plot([corner2[0], corner3[0]], [corner2[1], corner3[1]], 'k-', lw=1)

info_text = (f"c = (a·b)/(b·b) = {np.dot(a,b)}/{np.dot(b,b)} = {c:.4f}\n"
             f"e·b = {np.dot(error, b):.6f}  (≈ 0, confirms perpendicular)")
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Projection of a onto b')

plt.tight_layout()
plt.show()
