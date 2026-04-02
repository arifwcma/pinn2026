import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 500)

initial_condition = x**2 * np.cos(np.pi * x)

reaction_force = 5 * initial_condition**3 - 5 * initial_condition

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

axes[0].plot(x, initial_condition, 'b-', linewidth=2.5, label='u(0, x) = x² cos(πx)')
axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Phase A (u = +1)')
axes[0].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Phase B (u = −1)')
axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('u(0, x)', fontsize=12)
axes[0].set_title('Allen-Cahn: Initial condition at t = 0 (smooth, gentle curve)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].set_ylim(-1.5, 1.5)

final_u = np.tanh(x / (2 * np.sqrt(0.0001)))
final_u = np.where(x < -0.7, -1.0, final_u)
final_u = np.where(x > 0.7, -1.0, final_u)

sharp_transition_centers = [-0.7, 0.0, 0.7]
final_approx = np.ones_like(x) * (-1.0)
mask_inner = (x > -0.7) & (x < 0.0)
final_approx[mask_inner] = -1.0
mask_mid = (x > -0.05) & (x < 0.05)
final_approx[mask_mid] = np.tanh(x[mask_mid] / 0.01)
mask_right = (x > 0.0) & (x < 0.7)
final_approx[mask_right] = 1.0

width = 0.02
final_realistic = -1.0 * np.ones_like(x)
final_realistic += 2.0 * 0.5 * (1 + np.tanh((x + 0.7) / width))
final_realistic -= 2.0 * 0.5 * (1 + np.tanh((x - 0.0) / width))
final_realistic += 2.0 * 0.5 * (1 + np.tanh((x - 0.0) / width))
final_realistic -= 2.0 * 0.5 * (1 + np.tanh((x - 0.7) / width))

final_realistic = -1.0 * np.ones_like(x)
final_realistic += 2.0 / (1 + np.exp(-(x + 0.7) / width))
final_realistic -= 2.0 / (1 + np.exp(-(x + 0.0) / width))
final_realistic += 2.0 / (1 + np.exp(-(x - 0.0) / width))
final_realistic -= 2.0 / (1 + np.exp(-(x - 0.7) / width))

axes[1].plot(x, final_realistic, 'b-', linewidth=2.5, label='u(0.9, x) ≈ sharp step function')
axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Phase A (u = +1)')
axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Phase B (u = −1)')
axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('u(0.9, x)', fontsize=12)
axes[1].set_title('Allen-Cahn: Solution at t = 0.9 (nearly discontinuous — phases have separated)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].set_ylim(-1.5, 1.5)

axes[1].annotate('Phase B\n(u = −1)', xy=(-0.85, -1.0), fontsize=10, color='red',
                 fontweight='bold', ha='center', va='top')
axes[1].annotate('Phase A\n(u = +1)', xy=(-0.35, 1.0), fontsize=10, color='green',
                 fontweight='bold', ha='center', va='bottom')
axes[1].annotate('Phase B\n(u = −1)', xy=(0.85, -1.0), fontsize=10, color='red',
                 fontweight='bold', ha='center', va='top')

fig.suptitle('R138: Allen-Cahn — Smooth start → Sharp phase boundaries\n'
             'The PINN gets data at t=0.1 and must predict the nearly discontinuous solution at t=0.9',
             fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('demos/r138_allen_cahn_visualize.png', dpi=150, bbox_inches='tight')
plt.show()
