# Fundamentals → Runge-Kutta Learning Path

## Current state: COMPLETED

All RK fundamentals covered. Discrete time PINN models (Allen-Cahn, KdV) completed.

## Path to ENTER_RK

1. [x] Derivative = slope at a point (R93)
2. [x] Linear approximation: u(t+Δt) ≈ u(t) + Δt·u'(t) — and why it fails for large Δt (R93, R94 plot)
3. [x] Second derivative = how the slope itself changes (R95)
4. [x] Quadratic approximation is better than linear (R95 plot)
5. [x] Taylor series: what it is, why it works, the general pattern (R97–R108)
   - Derived from repeated application of fundamental theorem of calculus
   - Each unfolding peels off one term: s^n/n! · f^(n)(0)
6. [x] Why Taylor series is impractical for real ODE problems (R109, R112)
   - Need u'', u''' etc. which require computing F', F'' — explodes in complexity
7. [x] The RK idea: achieve Taylor-level accuracy using only F evaluations at clever points (R112, R114)
8. [x] RK1 (Euler) — one slope, first-order (R114)
9. [x] RK2 — two slopes, second-order, derive the weights (R115–R123, plot R120)
10. [x] RK3 — three slopes, third-order (R127–R128)
11. [x] RK4 — four slopes, the classic method (R129–R130)
12. [x] General RK: q stages, Butcher tableau (R131–R135)

## After ENTER_RK

13. [x] Explicit vs Implicit RK — what's the difference (R135)
14. [x] Why implicit matters (stability for stiff problems) (R135)
15. [x] How PINN paper uses implicit RK with hundreds of stages (R136)
16. [x] Back to the paper: Allen-Cahn (discrete forward) and KdV (discrete inverse) (R138–R213)
