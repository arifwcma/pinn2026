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

### Where to resume:
- [ ] Explain periodic boundary conditions (what "wrapping around" means concretely)
- [ ] Explain how complex-valued solutions are handled by the neural network (two outputs: real + imaginary)
- [ ] Then the Schrödinger PINN loss terms in detail
- [ ] Remaining topics from the BFS plan (Topics 3–9 above)

### Communication conventions:
- Every response is tagged R1, R2, R3... so learner can refer back
- Last response was R19
- Use $...$ and $$...$$ for math (NOT \\( \\) — those don't render in Cursor)
- Use Unicode Σ, ², ᵢ etc. as fallback if LaTeX fails

## Coding Standards

1. **Readability is the #1 objective.** Code must be instantly understandable.
2. Use descriptive variable names — no abbreviations unless universally obvious.
3. Break complex one-liners into multiple lines with intermediate variables.
4. No comments (per instructions.md).
5. No showing off short/clever code. Clarity over brevity, always.

## How to Continue

Tell the next agent:

> Read `.cursor/pinn-learning-plan.md` for context on what we are doing and where we left off. Then continue with the next unchecked topic in the plan. Follow the communication preferences in `.cursor/instructions.md`. The paper PDF is at `read/pinn_pdf/1-s2.0-S0021999118307125-am.pdf`.
