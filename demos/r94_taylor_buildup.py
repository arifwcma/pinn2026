import numpy as np
import matplotlib.pyplot as plt


def true_function(t):
    return np.sin(t) + 2


def true_derivative(t):
    return np.cos(t)


KNOWN_POINT = 1.0
DELTA_T = 1.5
PREDICTION_POINT = KNOWN_POINT + DELTA_T

t_curve = np.linspace(0, 4, 500)
u_curve = true_function(t_curve)

u_at_known = true_function(KNOWN_POINT)
slope_at_known = true_derivative(KNOWN_POINT)

t_tangent = np.linspace(KNOWN_POINT - 0.5, PREDICTION_POINT + 0.3, 100)
u_tangent = u_at_known + slope_at_known * (t_tangent - KNOWN_POINT)

u_linear_prediction = u_at_known + DELTA_T * slope_at_known
u_true_at_prediction = true_function(PREDICTION_POINT)
error = abs(u_true_at_prediction - u_linear_prediction)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_curve, u_curve, 'k-', linewidth=2, label='True curve u(t)')
ax.plot(t_tangent, u_tangent, 'r--', linewidth=2, label='Linear approx (tangent line)')

ax.plot(KNOWN_POINT, u_at_known, 'bo', markersize=10, zorder=5)
ax.annotate(
    f'Known: u({KNOWN_POINT}) = {u_at_known:.2f}\nSlope: u\'({KNOWN_POINT}) = {slope_at_known:.2f}',
    xy=(KNOWN_POINT, u_at_known),
    xytext=(KNOWN_POINT - 0.8, u_at_known - 0.6),
    fontsize=10, color='blue',
    arrowprops=dict(arrowstyle='->', color='blue')
)

ax.plot(PREDICTION_POINT, u_linear_prediction, 'rs', markersize=10, zorder=5)
ax.annotate(
    f'Linear prediction: {u_linear_prediction:.2f}',
    xy=(PREDICTION_POINT, u_linear_prediction),
    xytext=(PREDICTION_POINT + 0.15, u_linear_prediction + 0.2),
    fontsize=10, color='red'
)

ax.plot(PREDICTION_POINT, u_true_at_prediction, 'go', markersize=10, zorder=5)
ax.annotate(
    f'True value: {u_true_at_prediction:.2f}',
    xy=(PREDICTION_POINT, u_true_at_prediction),
    xytext=(PREDICTION_POINT + 0.15, u_true_at_prediction - 0.3),
    fontsize=10, color='green'
)

ax.annotate(
    '', xy=(PREDICTION_POINT + 0.05, u_true_at_prediction),
    xytext=(PREDICTION_POINT + 0.05, u_linear_prediction),
    arrowprops=dict(arrowstyle='<->', color='purple', lw=2)
)
ax.text(
    PREDICTION_POINT + 0.15,
    (u_true_at_prediction + u_linear_prediction) / 2,
    f'Error = {error:.2f}',
    fontsize=11, color='purple', fontweight='bold'
)

ax.axvline(x=KNOWN_POINT, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(x=PREDICTION_POINT, color='gray', linewidth=0.5, linestyle=':')
ax.text(KNOWN_POINT, ax.get_ylim()[0] + 0.05, f't_n = {KNOWN_POINT}', ha='center', fontsize=9, color='gray')
ax.text(PREDICTION_POINT, ax.get_ylim()[0] + 0.05, f't_n + Δt = {PREDICTION_POINT}', ha='center', fontsize=9, color='gray')

ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('u(t)', fontsize=12)
ax.set_title('R94: Linear approximation (Euler) — tangent line vs true curve', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('demos/r94_taylor_buildup.png', dpi=150)
plt.show()
