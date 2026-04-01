import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 1000)

def approximate_solution(t, x):
    initial = x**2 * np.cos(np.pi * x)
    sharpness = 1.0 + t * 80
    plateau = np.tanh(t * 5)

    result = np.zeros_like(x)
    for xi in range(len(x)):
        raw = initial[xi]
        pushed = raw + plateau * np.sign(raw) * (1.0 - abs(raw))
        pushed = np.clip(pushed, -1, 1)
        result[xi] = pushed
    return result

def better_approximate(t, x):
    width = max(0.15 * (1 - t * 1.0), 0.015)
    left_wall = -0.7 + 0.02 * t
    right_wall = 0.7 - 0.02 * t
    blend = min(t / 0.3, 1.0)

    initial = x**2 * np.cos(np.pi * x)
    sharp = -1.0 * np.ones_like(x)
    sharp += 2.0 / (1 + np.exp(-(x - left_wall) / width))
    sharp -= 2.0 / (1 + np.exp(-(x - right_wall) / width))

    return (1 - blend) * initial + blend * sharp


time_snapshots = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#000000']
linewidths = [1.5, 1.5, 1.8, 2.0, 2.2, 2.8]

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

for time_val, color, lw in zip(time_snapshots, colors, linewidths):
    if time_val == 0.0:
        u_snapshot = x**2 * np.cos(np.pi * x)
    else:
        u_snapshot = better_approximate(time_val, x)
    axes[0].plot(x, u_snapshot, color=color, linewidth=lw,
                 label=f't = {time_val:.1f}')

axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.3)
axes[0].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('u(t, x)', fontsize=12)
axes[0].set_title('Allen-Cahn: how the solution evolves over time\n'
                   'smooth curve gradually sharpens into step function',
                   fontsize=13)
axes[0].legend(fontsize=11, loc='upper left', ncol=2)
axes[0].set_ylim(-1.5, 1.5)

picked_x_values = [-0.9, -0.5, 0.0, 0.35, 0.5, 0.85]
picked_colors = ['#E91E63', '#9C27B0', '#607D8B', '#2196F3', '#4CAF50', '#FF5722']
time_fine = np.linspace(0, 0.9, 200)

for x_val, color in zip(picked_x_values, picked_colors):
    x_arr = np.array([x_val])
    u_over_time = []
    for t_val in time_fine:
        if t_val == 0.0:
            u_val = x_val**2 * np.cos(np.pi * x_val)
        else:
            u_val = better_approximate(t_val, x_arr)[0]
        u_over_time.append(u_val)

    axes[1].plot(time_fine, u_over_time, color=color, linewidth=2.0,
                 label=f'x = {x_val}')

axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.3)
axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
axes[1].set_xlabel('t (time)', fontsize=12)
axes[1].set_ylabel('u(t, x) at fixed x', fontsize=12)
axes[1].set_title('Allen-Cahn: tracking individual points over time\n'
                   'each point commits to +1 or −1',
                   fontsize=13)
axes[1].legend(fontsize=10, loc='center left', ncol=2)
axes[1].set_ylim(-1.5, 1.5)

fig.suptitle('R142: Allen-Cahn — the equation we are trying to model\n'
             'u_t = 0.0001·u_xx − 5u³ + 5u   on x ∈ [−1, 1]',
             fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('demos/r142_allen_cahn_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
