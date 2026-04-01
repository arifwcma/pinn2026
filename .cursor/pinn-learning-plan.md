# PINN Paper Study – Context & Plan

## Project Purpose

This project is a structured tutoring session to deeply understand the paper:
**"Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations"** by Maziar Raissi, Perdikaris, and Karniadakis (2019, Journal of Computational Physics).

The paper is available in three formats in `read/` (HTML, MHTML, PDF).

## Learner Profile

- Name: Arif (prefers Boss or Ostad)
- PhD in ML (hyperspectral remote sensing, band selection, dimensionality reduction)
- 16–18 years software engineering (Java, Python, etc.)
- Strong Python, applied ML, GIS, remote sensing
- Needs gut-level intuitive understanding, not surface-level
- Prefers hierarchical learning: high-level overview first, then BFS exploration
- Learns well from small code demos where applicable
- Wants brief, focused prompts — one small step at a time
- See `.cursor/instructions.md` for full communication preferences

## High-Level Ideas of the Paper (Completed)

We established six high-level pillars:

1. **Core insight**: Embed known physics (PDE) directly into the neural network's loss function, so the network can only learn physically consistent solutions.
2. **Two problem classes**:
   - Forward (data-driven solution): Know the PDE, have sparse initial/boundary data, want the full solution.
   - Inverse (data-driven discovery): Have scattered observations, want to discover unknown PDE parameters.
3. **Two model architectures per problem class**:
   - Continuous time: PDE enforced at random collocation points across the domain.
   - Discrete time: PDE encoded via Runge-Kutta time-stepping schemes.
4. **Secret weapon — automatic differentiation**: Backprop machinery reused to compute derivatives w.r.t. input coordinates (space, time), giving PDE residuals for free.
5. **Why tiny data works**: Physics constraint acts as a powerful regularizer, shrinking the solution space dramatically.
6. **Demonstrations**: Burgers', Schrödinger, Allen-Cahn, Navier-Stokes, KdV equations.

## Learning Plan (BFS Exploration Order)

Each topic below should be explored one at a time, with intuitive explanations and code demos where useful.

- [ ] **Topic 1**: The core insight — how physics gets embedded into the loss function. Walk through the Burgers' equation example with code.
- [ ] **Topic 2**: Forward problem (data-driven solution) — continuous time model. What are collocation points? How does MSE_u + MSE_f work?
- [ ] **Topic 3**: Forward problem — discrete time model. How Runge-Kutta stages get encoded into the neural network architecture.
- [ ] **Topic 4**: Inverse problem (data-driven discovery) — continuous time. How PDE parameters (λ) become trainable. Navier-Stokes example.
- [ ] **Topic 5**: Inverse problem — discrete time. Two-snapshot approach. KdV equation example.
- [ ] **Topic 6**: Automatic differentiation mechanics — how tf.gradients w.r.t. inputs differs from w.r.t. weights.
- [ ] **Topic 7**: Why physics-informed loss acts as regularization — connection to Bayesian priors and the small-data regime.
- [ ] **Topic 8**: Practical details — network architecture choices, L-BFGS optimizer, collocation point sampling (Latin Hypercube), sensitivity to hyperparameters.
- [ ] **Topic 9**: Hands-on — build a minimal PINN from scratch in Python/PyTorch for Burgers' equation.

## Session 1 Progress (R1–R19)

### What was covered:
- [x] High-level overview delivered (all 6 pillars)
- [x] Forward vs Inverse understood via ball toy example (R1–R7)
  - Forward: know the equation, learn the solution (demos/r1_forward_ball.py)
  - Inverse: have observations, learn unknown parameters (demos/r2_inverse_ball.py)
  - Key difference in code: `nn.Parameter` for trainable gravity vs constant
- [x] Core PINN mechanics understood: loss = loss_data + loss_physics
  - Physics residual = plug network's prediction into the PDE, measure how far from zero
  - autograd.grad computes derivatives w.r.t. inputs (not weights) — treat `create_graph=True` and `grad_outputs=ones` as PyTorch boilerplate
- [x] Burgers' equation forward demo (R9–R11, demos/r10_burgers_forward.py)
  - Jump from ODE (1 input) to PDE (2 inputs: t, x)
  - Three autograd calls: u_t, u_x, u_xx
- [x] General PDE template from the paper: u_t + N[u; λ] = 0 (R13–R14)
  - N is a nonlinear operator (calligraphic N), different for each equation
  - Every example in the paper fits this template
- [x] Paper structure: continuous/discrete × forward/inverse grid confirmed (R15)
  - Continuous forward: Schrödinger | Discrete forward: Allen-Cahn
  - Continuous inverse: Navier-Stokes | Discrete inverse: KdV
  - Burgers' used in appendices for systematic studies across all four
- [x] Started Schrödinger deep-dive (R16–R19)
  - Equation describes wave evolution: spreading vs self-focusing
  - Three loss terms: loss_initial + loss_boundary + loss_physics
  - Periodic boundary conditions and complex-valued output introduced but NOT yet explained
  - Created demos/s1_schrodinger_visualize.py to visualize the wave evolution

## Session 2 Progress (R20–R85)

### What was covered:
- [x] Complex-valued solutions in PINNs (R20–R21, R33–R36)
  - Network outputs two real values (u, v) instead of one complex h
  - Schrödinger equation split into two real residuals
  - Key insight: if a + ib = 0, then a = 0 AND b = 0 — this makes the split valid
- [x] Schrödinger equation deep-dive completed (R22–R39)
  - Equation describes wave evolution: spreading vs self-focusing
  - Initial condition: h(0,x) = 2sech(x) — bell-shaped bump
  - Visualizations: s1_schrodinger_visualize.py, s1_supplementary1.py, s1_sch_vis2.py (animation), s1_sch_vis3.py (complex plane)
  - Periodic boundary conditions: left edge = right edge in value and slope (R39)
  - Three loss terms: loss_initial + loss_boundary + loss_physics
  - Full architecture diagram: demos/r37_schrodinger_architecture.py
- [x] Navier-Stokes inverse problem deep-dive (R44–R84)
  - Physical setup: 2D flow past a cylinder, Kármán vortex street (R45–R46, R62)
  - Three equations: two momentum + incompressibility u_x + v_y = 0 (R46 rewritten)
  - Incompressibility = mass conservation explained at gut level (R47–R60)
  - Stream function trick: learn ψ instead of (u,v), derive u = ψ_y and v = -ψ_x (R68–R75)
    - Guarantees incompressibility exactly, not approximately
    - ψ is a mathematical invention — defined such that its derivatives give velocities
  - Data: 5,000 scattered velocity measurements (1% of full data), zero pressure data (R61, R64, R66)
  - Goal 1: discover unknown λ₁ and λ₂ (R63, R67)
  - Goal 2: reconstruct entire pressure field without any pressure measurements (R80–R84)
    - Possible because Navier-Stokes equations link pressure and velocity
    - Pressure unique up to a constant (R84)
  - Loss: loss_data (velocity mismatch) + loss_physics (two momentum residuals) (R77)
  - Architecture diagram: demos/r78_navier_stokes_architecture.py
  - Collocation points explained: points where PDE residual is checked (R78)

### Communication conventions:
- Every response is tagged R1, R2, R3... so learner can refer back
- Last response was R213 (Session 4 ended here)
- Use $...$ and $$...$$ for math (NOT \\( \\) — those don't render in Cursor)
- \\sqrt does NOT render in Cursor — use \\surd instead
- Use Unicode Σ, ², ᵢ etc. as fallback if LaTeX fails
- Never rush the learner — stay with a concept as long as needed
- Keep responses short — one chunk at a time, ask before continuing
- When introducing math quantities (like k1, k2), always follow the formula with a plain-English one-liner explaining what it physically means
- **CRITICAL**: Never introduce new symbols without immediately saying what they mean in plain English. Always qualify "predicted" — say "network-guessed u" vs "RK-predicted u_n". The learner gets very frustrated by unexplained notation.

## Coding Standards

1. **Readability is the #1 objective.** Code must be instantly understandable.
2. Use descriptive variable names — no abbreviations unless universally obvious.
3. Break complex one-liners into multiple lines with intermediate variables.
4. No comments (per instructions.md).
5. No showing off short/clever code. Clarity over brevity, always.

## Session 3 Progress (R86–R125)

### What was covered:
- [x] Runge-Kutta fundamentals — bottom-up from Taylor series (R93–R125)
  - Taylor series derivation from first principles using fundamental theorem of calculus (R97–R108)
  - Why Taylor is impractical for real ODEs: need higher derivatives of F (R109, R112)
  - The RK idea: achieve Taylor-level accuracy using only F evaluations at clever points (R112, R114)
  - RK1 (Euler): 1 stage, 1st order — just one slope at the start (R114)
  - RK2: 2 stages, 2nd order — sample slope at start AND at a trial point (R115–R125)
    - k1 = Δt · F(tₙ, uₙ) = "how much u would change using starting slope for a full step"
    - k2 = Δt · F(tₙ + αΔt, uₙ + αk1) = "how much u would change using trial-position slope for a full step"
    - Weights: ½k1 + ½k2 is one common choice (not the only one)
  - Visualization: demos/r120_rk2_visualize.py — shows slopes, trial point, Euler vs RK2 vs true
  - Visualization: demos/r94_taylor_buildup.py — linear approximation and error
  - Visualization: demos/r95_taylor_second_order.py — 1st vs 2nd order Taylor comparison

## Session 4 Progress (R126–R213)

### What was covered:
- [x] RK3 — three slopes, third-order (R127–R128)
  - k3's strange starting position u_n - k1 + 2k2 explained: algebra forces a=-1, b=2 to match Taylor (R128)
  - "Undo naive start-slope, double-down on midpoint information"
- [x] RK4 — the classic four-slope method (R129–R130)
  - Weights 1/6, 2/6, 2/6, 1/6 (Simpson's pattern)
  - Two midpoint samples (k2 and k3) — midpoint has most curvature info
  - RK4 has enough stages that each can do simpler jumps (no weird coefficients)
- [x] RK-N are independent families, not extensions of each other (R131–R132)
  - Practical sweet spot: order 4–5 (ode45)
  - Beyond order 8, cost outweighs benefit
- [x] q (stages) vs N (order) distinction — q can be much larger than order (R133–R134)
- [x] Butcher tableau — unified notation for all RK methods (R135)
  - Explicit: lower triangular A matrix
  - Implicit: full A matrix — creates chicken-and-egg coupling
- [x] Explicit vs Implicit RK (R135)
  - Explicit: forward sweep, O(q) F-evaluations
  - Implicit: solve coupled system, O(q³d³) per Newton iteration — polynomial, not exponential
- [x] PINN paper's implicit RK trick (R136)
  - Neural network OUTPUTS the stage values directly — no system to solve
  - Cost of more stages = just more output neurons (practically free)
  - Enables q=100 or q=500 — unprecedented in classical methods
- [x] Allen-Cahn equation deep-dive (R138–R203)
  - Phase separation: u→+1 (phase A) or u→-1 (phase B) (R138–R143)
  - Reaction term 5u³-5u pushes toward ±1, tiny diffusion smooths boundaries (R139)
  - Visualizations: r138_allen_cahn_visualize.py, r139_allen_cahn_reaction.py, r142_allen_cahn_evolution.py, r143_allen_cahn_animation.py
  - Why discrete not continuous: sharp transitions need dense collocation points, discrete avoids this (R144–R146)
  - Full architecture walkthrough with q=2 toy example (R147–R203)
    - Named layers: Input → Hidden NN → û Layer → Slope Layer → RK Layer → Loss Layer
    - û Layer: network-guessed u at future times (no ground truth to compare)
    - Slope Layer: u_t from the PDE (purple boxes in diagram)
    - RK Layer: RK-predicted u_n — subtract RK-blended slopes from guesses to get implied starting value (blue boxes)
    - Loss: compare RK-predicted u_n against actual known u_n at t=0.1
    - Key insight from learner: "build the computation graph so ground truth is at the end" (R196)
    - Boundary loss: periodic BCs — left edge = right edge in value and slope
  - Architecture diagrams: r150_allen_cahn_architecture.py, r152_allen_cahn_simple.py, r157_allen_cahn_clear.py, r167_allen_cahn_final.py
  - Concrete layer-by-layer walkthrough with 3 samples (R181–R203)
    - Symbols: û_i^j (network guess, stage i, sample j), s_i^j (slope), p_i^j (RK-predicted u_n)
  - Forward has q+1 outputs (need to predict u_{n+1}), inverse has q outputs (both ends have data)
- [x] KdV equation deep-dive (R206–R212)
  - Shallow water waves: u = water surface height (R206–R207)
  - Two competing terms: nonlinear steepening vs dispersion
  - Inverse problem: discover λ₁ and λ₂ from two data snapshots
  - Architecture: same as Allen-Cahn but (1) two snapshots, (2) λ₁, λ₂ are trainable nn.Parameters
  - RK Layer produces two sets: RK-predicted u_n (matches t=0.2 data) and RK-predicted u_{n+1} (matches t=0.8 data)
  - Both use same network-guessed stages — stages are the bridge connecting the two timestamps (R212)
  - Concrete layer-by-layer walkthrough with 3 samples (R210)
  - Architecture diagram: r208_kdv_architecture.py

### Lessons learned about communication:
- NEVER introduce new symbols (like r, N, f) without immediately defining them in plain English
- Always qualify "predicted": "network-guessed u" vs "RK-predicted u_n"
- When the learner is confused, make things simpler, not longer
- The learner's own verbalization is often the best summary — adopt it

### Remaining topics:
- [ ] Autograd mechanics — how differentiating w.r.t. inputs differs from w.r.t. weights
- [ ] Physics loss as regularization — why small data works, connection to Bayesian priors
- [ ] Practical details — L-BFGS, Latin Hypercube sampling, network sizing, sensitivity studies
- [ ] Hands-on capstone — build a complete PINN from scratch for one problem

## How to Continue

Tell the next agent:

> Read `.cursor/pinn-learning-plan.md` and `.cursor/rk-fundamentals-plan.md` for full context. Also read `.cursor/instructions.md` for communication preferences.
>
> Continue from where Session 4 left off. The learner has completed all four paper quadrants (Schrödinger, Navier-Stokes, Allen-Cahn, KdV) and the full RK path. Ready for **remaining topics**: autograd mechanics, physics-as-regularization, practical details, or hands-on capstone. Start numbering responses from **R214**.
>
> Key rules:
> 1. **Start every response with a tag** like R214, R215, etc.
> 2. Use `$...$` for inline math and `$$...$$` for block math. `\sqrt` does NOT render — use `\surd` instead.
> 3. When introducing math quantities, always follow each formula with a **plain-English one-liner** explaining what it physically means.
> 4. **NEVER introduce new symbols without immediately saying what they mean.** The learner gets very frustrated by unexplained notation. Always qualify "predicted" — say "network-guessed u" vs "RK-predicted u_n".
> 5. Never rush the learner. Stay with each concept as long as needed. One small chunk per response.
> 6. Keep responses brief. Ask before continuing to the next topic.
> 7. No comments in code. Descriptive variable names. Readability over cleverness.
> 8. The paper PDF is at `read/pinn_pdf/1-s2.0-S0021999118307125-am.pdf`.
> 9. Previous demos are in `demos/`.
> 10. The learner prefers Boss or Ostad. Address him respectfully.
> 11. When the learner is confused, make things **simpler**, not longer. Short sentences. Concrete examples.
