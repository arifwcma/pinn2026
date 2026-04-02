import numpy as np
import matplotlib.pyplot as plt


def true_function(t):
    return np.sin(t) + 2


def true_derivative(t):
    return np.cos(t)


def true_second_derivative(t):
    return -np.sin(t)


KNOWN_POINT = 1.0
DELTA_T = 1.5
PREDICTION_POINT = KNOWN_POINT + DELTA_T

t_curve = np.linspace(0, 4, 500)
u_curve = true_function(t_curve)

u_at_known = true_function(KNOWN_POINT)
slope_at_known = true_derivative(KNOWN_POINT)
curvature_at_known = true_second_derivative(KNOWN_POINT)

t_approx = np.linspace(KNOWN_POINT - 0.3, PREDICTION_POINT + 0.3, 200)

u_linear = u_at_known + slope_at_known * (t_approx - KNOWN_POINT)

u_quadratic = (u_at_known
               + slope_at_known * (t_approx - KNOWN_POINT)
               + 0.5 * curvature_at_known * (t_approx - KNOWN_POINT)**2)

linear_prediction = u_at_known + DELTA_T * slope_at_known
quadratic_prediction = (u_at_known
                        + DELTA_T * slope_at_known
                        + 0.5 * curvature_at_known * DELTA_T**2)
true_value = true_function(PREDICTION_POINT)

linear_error = abs(true_value - linear_prediction)
quadratic_error = abs(true_value - quadratic_prediction)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_curve, u_curve, 'k-', linewidth=2, label='True curve u(t)')
ax.plot(t_approx, u_linear, 'r--', linewidth=1.5, alpha=0.5, label=f'1st order (linear) — error = {linear_error:.3f}')
ax.plot(t_approx, u_quadratic, 'b--', linewidth=2, label=f'2nd order (quadratic) — error = {quadratic_error:.3f}')

ax.plot(KNOWN_POINT, u_at_known, 'ko', markersize=10, zorder=5)
ax.annotate(
    f'Known point\nu = {u_at_known:.2f}\nu\' = {slope_at_known:.2f}\nu\'\' = {curvature_at_known:.2f}',
    xy=(KNOWN_POINT, u_at_known),
    xytext=(KNOWN_POINT - 0.9, u_at_known - 0.8),
    fontsize=9, color='black',
    arrowprops=dict(arrowstyle='->', color='black')
)

ax.plot(PREDICTION_POINT, linear_prediction, 'rs', markersize=8, zorder=5)
ax.plot(PREDICTION_POINT, quadratic_prediction, 'bs', markersize=8, zorder=5)
ax.plot(PREDICTION_POINT, true_value, 'go', markersize=10, zorder=5)

ax.annotate(
    f'True: {true_value:.3f}',
    xy=(PREDICTION_POINT, true_value),
    xytext=(PREDICTION_POINT + 0.15, true_value + 0.15),
    fontsize=10, color='green', fontweight='bold'
)

ax.annotate(
    f'Linear: {linear_prediction:.3f}',
    xy=(PREDICTION_POINT, linear_prediction),
    xytext=(PREDICTION_POINT + 0.15, linear_prediction + 0.1),
    fontsize=9, color='red'
)

ax.annotate(
    f'Quadratic: {quadratic_prediction:.3f}',
    xy=(PREDICTION_POINT, quadratic_prediction),
    xytext=(PREDICTION_POINT + 0.15, quadratic_prediction - 0.2),
    fontsize=9, color='blue'
)

ax.axvline(x=KNOWN_POINT, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(x=PREDICTION_POINT, color='gray', linewidth=0.5, linestyle=':')
ax.text(KNOWN_POINT, ax.get_ylim()[0] + 0.05, f't_n', ha='center', fontsize=9, color='gray')
ax.text(PREDICTION_POINT, ax.get_ylim()[0] + 0.05, f't_n + Δt', ha='center', fontsize=9, color='gray')

ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('u(t)', fontsize=12)
ax.set_title('R95: 1st vs 2nd order Taylor approximation', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('demos/r95_taylor_second_order.png', dpi=150)
plt.show()
