# Instructions for the Next Agent

## Who Is the Learner

Arif. PhD in ML (hyperspectral remote sensing). Strong Python. Needs gut-level intuitive understanding of every concept — not surface-level. Read `.cursor/instructions.md` and `.cursor/project.md` for his full communication profile before doing anything.

## What Has Been Completed

Arif has completed a deep study of Gaussian Processes (GP) and the "Regression Bridge" connecting OLS → BLR → GP. All materials are archived in `archive/gp_study/`. The progress file at `archive/gp_study/gp-study-progress.md` has the full breakdown.

Key concepts Arif now owns:
1. Single and multivariate Gaussian distributions
2. Conditioning (observing variables → updating beliefs)
3. GP as a distribution over functions (RBF kernel, length_scale, prior/posterior)
4. GP regression with full numerical worked examples
5. OLS linear regression in matrix form (closed-form solution, proof of normal equation)
6. Bayesian Linear Regression (prior over weights, posterior update, sequential updates)
7. Bayes' theorem (concrete soil carbon analogy, rectangle diagrams)
8. OLS vs BLR vs GP comparison on house price data

## What to Do Next

Continue with the remaining topics from the GP mastery roadmap. Arif wants to start with **item 6**:

### 6. Linear Algebra Prerequisites for GP Mastery
- Eigenvalues and eigenvectors — what they mean geometrically, why they matter for covariance matrices
- Positive definiteness — why kernels must produce positive definite matrices, what breaks if they don't
- Cholesky decomposition — the "square root" of a matrix, used in efficient GP sampling
- Matrix inverse via eigendecomposition — connecting to the O(n³) cost of GP
- Rank and condition number — when matrices are "nearly singular" and what that means in practice

### After That (In Order)
1. Multi-sample GP worked example (3 soil samples, 2 bands → SOC prediction)
2. Kernel hyperparameter optimization (maximum likelihood estimation)
3. Other kernel types (linear, Matérn, periodic, combined)
4. GP for classification
5. Scalability solutions (sparse GP, inducing points)

## How Arif Learns Best
1. **Hierarchical**: High-level overview first, then BFS one concept at a time
2. **Intuitive**: Analogies for abstract concepts. Real examples. "Gut-level feel" before moving on
3. **Numerical**: He needs to see actual numbers. When explaining any formula, walk through it with a tiny concrete example (2×2 matrices, 2-3 data points)
4. **Code-first**: Every concept should be accompanied by a Python script with visualization
5. **LaTeX PDFs for proofs**: When showing derivations or lengthy math, generate a Python script that creates a `.tex` file, then compile it. Use `tcolorbox` for "gut feeling" callout boxes. Number every equation
6. **No comments in code**: Zero inline comments, zero block comments, zero docstrings
7. **Descriptive variable names**: `covariance_matrix` not `K`, `training_features` not `X`
8. **One step at a time**: Never rush to the next topic. Stay with the current concept until he signals readiness
9. **Brief responses**: Unless he asks for detail, keep responses short and direct
10. **Numbered lists**: Never bullet points for lists

## Tools and Environment
- Python 3.14 with numpy, pandas, matplotlib, scipy available (`.venv/`)
- LaTeX installed via basictex (`pdflatex` available, plus `amsmath`, `amssymb`, `tcolorbox`, `geometry` packages)
- macOS (darwin), zsh shell
- Cursor IDE

## Response Format Convention
- Start every response with a token like R1, R2, etc. (continuing from where the previous agent left off — the last token was approximately R87, so start from R88)
- Every equation must have a label or number
- Define every new symbol immediately in plain English
