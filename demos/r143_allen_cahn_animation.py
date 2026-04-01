import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.linspace(-1, 1, 1000)


def approximate_solution(t, x):
    width = max(0.15 * (1 - t * 1.0), 0.015)
    left_wall = -0.7 + 0.02 * t
    right_wall = 0.7 - 0.02 * t
    blend = min(t / 0.3, 1.0)

    initial = x**2 * np.cos(np.pi * x)
    sharp = -1.0 * np.ones_like(x)
    sharp += 2.0 / (1 + np.exp(-(x - left_wall) / width))
    sharp -= 2.0 / (1 + np.exp(-(x - right_wall) / width))

    return (1 - blend) * initial + blend * sharp


fig, ax = plt.subplots(figsize=(12, 6))

ax.axhline(y=1, color='green', linestyle='--', alpha=0.4, label='Phase A (u = +1)')
ax.axhline(y=-1, color='red', linestyle='--', alpha=0.4, label='Phase B (u = −1)')

line, = ax.plot([], [], 'b-', linewidth=2.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                    fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(t, x)', fontsize=12)
ax.set_title('R143: Allen-Cahn — phase separation in action\n'
             'smooth initial condition sharpens into step function',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')

time_values = np.linspace(0, 0.9, 120)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def update(frame):
    t = time_values[frame]
    if t == 0.0:
        u_snapshot = x**2 * np.cos(np.pi * x)
    else:
        u_snapshot = approximate_solution(t, x)
    line.set_data(x, u_snapshot)
    time_text.set_text(f't = {t:.3f}')
    return line, time_text


anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=len(time_values),
                                interval=60, blit=True, repeat=True)

plt.tight_layout()
plt.show()
