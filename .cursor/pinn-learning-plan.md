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

### Remaining topics:
- [ ] Discrete time models — Runge-Kutta refresher, then Allen-Cahn (forward) and KdV (inverse)
- [ ] Autograd mechanics — how differentiating w.r.t. inputs differs from w.r.t. weights
- [ ] Physics loss as regularization — why small data works, connection to Bayesian priors
- [ ] Practical details — L-BFGS, Latin Hypercube sampling, network sizing, sensitivity studies
- [ ] Hands-on capstone — build a complete PINN from scratch for one problem

### Communication conventions:
- Every response is tagged R1, R2, R3... so learner can refer back
- Last response was R85
- Use $...$ and $$...$$ for math (NOT \\( \\) — those don't render in Cursor)
- \\sqrt does NOT render in Cursor — use \\surd instead
- Use Unicode Σ, ², ᵢ etc. as fallback if LaTeX fails
- Never rush the learner — stay with a concept as long as needed
- Keep responses short — one chunk at a time, ask before continuing

## Coding Standards

1. **Readability is the #1 objective.** Code must be instantly understandable.
2. Use descriptive variable names — no abbreviations unless universally obvious.
3. Break complex one-liners into multiple lines with intermediate variables.
4. No comments (per instructions.md).
5. No showing off short/clever code. Clarity over brevity, always.

## How to Continue

Tell the next agent:

> Read `.cursor/pinn-learning-plan.md` for full context. Continue from where Session 2 left off — the next topic is discrete time models (Runge-Kutta refresher, then Allen-Cahn forward and KdV inverse). Start numbering responses from R86. Use `$...$` and `$$...$$` for math. `\sqrt` doesn't render — use `\surd` instead. Follow `.cursor/instructions.md` for communication preferences. Never rush — stay with each concept until the learner is satisfied. Keep responses short, one chunk at a time. The paper PDF is at `read/pinn_pdf/1-s2.0-S0021999118307125-am.pdf`.
