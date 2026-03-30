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

## Current Progress

- [x] High-level overview delivered (all 6 pillars)
- [ ] Topic 1 not yet started — this is where the next session should begin

## How to Continue

Tell the next agent:

> Read `.cursor/pinn-learning-plan.md` for context on what we are doing and where we left off. Then continue with the next unchecked topic in the plan. Follow the communication preferences in `.cursor/instructions.md`. The paper PDF is at `read/pinn_pdf/1-s2.0-S0021999118307125-am.pdf`.
