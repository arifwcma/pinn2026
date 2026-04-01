# Fundamentals → Runge-Kutta Learning Path

## Current state: FUNDAMENTAL_STATE

We are building up from basics. No concept is assumed known unless explicitly covered.

## Path to ENTER_RK

1. [x] Derivative = slope at a point (R93)
2. [x] Linear approximation: u(t+Δt) ≈ u(t) + Δt·u'(t) — and why it fails for large Δt (R93, R94 plot)
3. [x] Second derivative = how the slope itself changes (R95)
4. [x] Quadratic approximation is better than linear (R95 plot)
5. [x] Taylor series: what it is, why it works, the general pattern (R97–R108)
   - Derived from repeated application of fundamental theorem of calculus
   - Each unfolding peels off one term: s^n/n! · f^(n)(0)
6. [ ] Why Taylor series is impractical for real ODE problems (need u'', u''' etc.)
7. [ ] The RK idea: achieve Taylor-level accuracy using only F evaluations at clever points
8. [ ] RK1 (Euler) — one slope, first-order
9. [ ] RK2 — two slopes, second-order, derive the weights
10. [ ] RK4 — four slopes, the classic method
11. [ ] General RK: q stages, Butcher tableau

## After ENTER_RK

12. [ ] Explicit vs Implicit RK — what's the difference
13. [ ] Why implicit matters (stability for stiff problems)
14. [ ] How PINN paper uses implicit RK with hundreds of stages
15. [ ] Back to the paper: Allen-Cahn (discrete forward) and KdV (discrete inverse)
