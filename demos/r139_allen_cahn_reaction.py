import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(-1.5, 1.5, 500)

reaction_term = 5 * u**3 - 5 * u

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(u, reaction_term, 'b-', linewidth=2.5)
axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)

axes[0].plot(-1, 0, 'go', markersize=14, zorder=5)
axes[0].annotate('STABLE\nu = −1 (Phase B)', xy=(-1, 0),
                 xytext=(-1.4, 15), fontsize=10, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green'))

axes[0].plot(1, 0, 'go', markersize=14, zorder=5)
axes[0].annotate('STABLE\nu = +1 (Phase A)', xy=(1, 0),
                 xytext=(0.5, -20), fontsize=10, color='green', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='green'))

axes[0].plot(0, 0, 'rs', markersize=12, zorder=5)
axes[0].annotate('UNSTABLE\nu = 0', xy=(0, 0),
                 xytext=(0.2, 10), fontsize=10, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'))

axes[0].annotate('', xy=(-0.8, 0), xytext=(-0.3, 0),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
axes[0].text(-0.55, -1.5, 'pushed\ntoward −1', fontsize=9, color='purple', ha='center')

axes[0].annotate('', xy=(0.8, 0), xytext=(0.3, 0),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
axes[0].text(0.55, -1.5, 'pushed\ntoward +1', fontsize=9, color='purple', ha='center')

axes[0].set_xlabel('u', fontsize=12)
axes[0].set_ylabel('5u³ − 5u (reaction force)', fontsize=12)
axes[0].set_title('The reaction term: pushes u toward ±1', fontsize=13)
axes[0].set_ylim(-25, 25)

x = np.linspace(-1, 1, 500)
initial_condition = x**2 * np.cos(np.pi * x)
axes[1].plot(x, initial_condition, 'b-', linewidth=2.5, label='u(0, x) = x² cos(πx)')
axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.5)
axes[1].axhline(y=-1, color='green', linestyle='--', alpha=0.5)
axes[1].axhline(y=0, color='red', linestyle=':', alpha=0.5)

for sample_x in np.linspace(-0.9, 0.9, 15):
    sample_u = sample_x**2 * np.cos(np.pi * sample_x)
    if sample_u > 0.01:
        axes[1].annotate('', xy=(sample_x, min(sample_u + 0.15, 1.0)),
                         xytext=(sample_x, sample_u),
                         arrowprops=dict(arrowstyle='->', color='green', alpha=0.6, lw=1.5))
    elif sample_u < -0.01:
        axes[1].annotate('', xy=(sample_x, max(sample_u - 0.15, -1.0)),
                         xytext=(sample_x, sample_u),
                         arrowprops=dict(arrowstyle='->', color='green', alpha=0.6, lw=1.5))

axes[1].text(0.5, 1.1, 'u = +1 (attractor)', fontsize=10, color='green', ha='center')
axes[1].text(0.5, -1.15, 'u = −1 (attractor)', fontsize=10, color='green', ha='center')
axes[1].text(0.5, 0.07, 'u = 0 (unstable)', fontsize=9, color='red', ha='center')

axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('u(0, x)', fontsize=12)
axes[1].set_title('Initial condition with arrows showing where reaction pushes', fontsize=13)
axes[1].set_ylim(-1.5, 1.5)
axes[1].legend(fontsize=11, loc='upper left')

fig.suptitle('R139: Allen-Cahn equation — the reaction term creates two competing phases\n'
             'u_t = 0.0001·u_xx − 5u³ + 5u   (tiny diffusion + strong reaction)',
             fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('demos/r139_allen_cahn_reaction.png', dpi=150, bbox_inches='tight')
plt.show()
