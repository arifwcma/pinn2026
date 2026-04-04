import numpy as np
import matplotlib.pyplot as plt

original_vector = np.array([1, 2])

transformation_matrix = np.array([
    [2, 0],
    [0, 2]
])

transformed_vector = transformation_matrix @ original_vector

fig, ax = plt.subplots(figsize=(6, 6))

ax.quiver(0, 0, original_vector[0], original_vector[1],
           angles='xy', scale_units='xy', scale=1, color='blue', label='v (original)')

ax.quiver(0, 0, transformed_vector[0], transformed_vector[1],
           angles='xy', scale_units='xy', scale=1, color='red', label="v' (transformed)")

all_coords = np.array([original_vector, transformed_vector, [0, 0]])
margin = 1
ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend()
ax.set_title(f"Matrix:\n{transformation_matrix}")

plt.tight_layout()
plt.show()
