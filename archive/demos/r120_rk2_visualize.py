import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def slope_function(t, u):
    return -0.5 * u + np.sin(t)


KNOWN_TIME = 0.5
DELTA_T = 1.5
ALPHA = 0.5
PREDICTION_TIME = KNOWN_TIME + DELTA_T

exact_solution = solve_ivp(slope_function, [0, 4], [2.0], t_eval=np.linspace(0, 4, 500), max_step=0.01)
t_curve = exact_solution.t
u_curve = exact_solution.y[0]

u_n = np.interp(KNOWN_TIME, t_curve, u_curve)
true_value = np.interp(PREDICTION_TIME, t_curve, u_curve)

starting_slope = slope_function(KNOWN_TIME, u_n)
k1 = DELTA_T * starting_slope

trial_time = KNOWN_TIME + ALPHA * DELTA_T
trial_u = u_n + ALPHA * k1
trial_slope = slope_function(trial_time, trial_u)
k2 = DELTA_T * trial_slope

euler_prediction = u_n + k1
rk2_prediction = u_n + 0.5 * k1 + 0.5 * k2

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(t_curve, u_curve, 'k-', linewidth=2.5, label='True curve u(t)')

ax.plot(KNOWN_TIME, u_n, 'ko', markersize=12, zorder=5)
ax.annotate(
    f'START\nu_n = {u_n:.2f}\nslope = {starting_slope:.2f}',
    xy=(KNOWN_TIME, u_n),
    xytext=(KNOWN_TIME - 0.45, u_n + 0.35),
    fontsize=10, color='black', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='black')
)

t_k1_line = np.array([KNOWN_TIME, PREDICTION_TIME])
u_k1_line = np.array([u_n, u_n + k1])
ax.plot(t_k1_line, u_k1_line, 'r-', linewidth=2, alpha=0.7)
ax.plot(PREDICTION_TIME, euler_prediction, 'rs', markersize=12, zorder=5)
ax.annotate(
    f'EULER (k1 only)\nu = {euler_prediction:.2f}\n'
    f'k1 = {DELTA_T} × {starting_slope:.2f} = {k1:.2f}',
    xy=(PREDICTION_TIME, euler_prediction),
    xytext=(PREDICTION_TIME + 0.1, euler_prediction + 0.2),
    fontsize=9, color='red'
)

ax.plot(trial_time, trial_u, 'mD', markersize=12, zorder=5)
ax.annotate(
    f'TRIAL POSITION (peek ahead)\n'
    f'u_trial = u_n + α·k1\n'
    f'= {u_n:.2f} + {ALPHA}×{k1:.2f} = {trial_u:.2f}\n'
    f'slope here = {trial_slope:.2f}',
    xy=(trial_time, trial_u),
    xytext=(trial_time - 1.0, trial_u + 0.5),
    fontsize=9, color='purple',
    arrowprops=dict(arrowstyle='->', color='purple')
)

t_k2_visual = np.array([KNOWN_TIME, PREDICTION_TIME])
u_k2_visual = np.array([u_n, u_n + k2])
ax.plot(t_k2_visual, u_k2_visual, 'm--', linewidth=2, alpha=0.7)

ax.plot(PREDICTION_TIME, rk2_prediction, 'b^', markersize=14, zorder=5)
ax.annotate(
    f'RK2 (blend two slopes)\n'
    f'u = u_n + ½·k1 + ½·k2\n'
    f'= {u_n:.2f} + ½×{k1:.2f} + ½×{k2:.2f}\n'
    f'= {rk2_prediction:.2f}',
    xy=(PREDICTION_TIME, rk2_prediction),
    xytext=(PREDICTION_TIME + 0.1, rk2_prediction - 0.55),
    fontsize=9, color='blue',
    arrowprops=dict(arrowstyle='->', color='blue')
)

ax.plot(PREDICTION_TIME, true_value, 'go', markersize=14, zorder=5)
ax.annotate(
    f'TRUE VALUE\n{true_value:.2f}',
    xy=(PREDICTION_TIME, true_value),
    xytext=(PREDICTION_TIME + 0.1, true_value + 0.15),
    fontsize=10, color='green', fontweight='bold'
)

euler_error = abs(true_value - euler_prediction)
rk2_error = abs(true_value - rk2_prediction)
ax.text(0.1, ax.get_ylim()[1] - 0.3,
        f'Euler error: {euler_error:.3f}\nRK2 error:   {rk2_error:.3f}',
        fontsize=12, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

ax.axvline(x=KNOWN_TIME, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(x=trial_time, color='purple', linewidth=0.5, linestyle=':')
ax.axvline(x=PREDICTION_TIME, color='gray', linewidth=0.5, linestyle=':')
ax.text(KNOWN_TIME, ax.get_ylim()[0] + 0.02, 't_n', ha='center', fontsize=9, color='gray')
ax.text(trial_time, ax.get_ylim()[0] + 0.02, 't_n + α·Δt', ha='center', fontsize=9, color='purple')
ax.text(PREDICTION_TIME, ax.get_ylim()[0] + 0.02, 't_n + Δt', ha='center', fontsize=9, color='gray')

ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('u(t)', fontsize=12)
ax.set_title('R120: RK2 — Peek ahead, get a second slope, blend them', fontsize=13)
ax.legend(fontsize=11, loc='lower left')
plt.tight_layout()
plt.savefig('demos/r120_rk2_visualize.png', dpi=150)
plt.show()
