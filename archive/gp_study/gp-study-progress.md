# GP Study — Progress Summary

## What Was Covered

### 1. Gaussian Distribution Fundamentals
- Single Gaussian: mean, variance, bell curve shape
- Central Limit Theorem — why Gaussian appears often but is not universal
- GP models uncertainty about the function, not the data's distribution

### 2. Multivariate Gaussian
- Mean vector, covariance matrix
- Positive, zero, negative covariance — visualized with scatter plots
- Covariance = "how much do these variables know about each other"

### 3. Conditioning (The Superpower)
- Observing one variable updates belief about others
- Mean shifts, variance shrinks
- Prior → posterior terminology introduced
- Posterior is always at least as specified as the prior

### 4. What a Gaussian Process Is
- A multivariate Gaussian scaled to cover a whole function
- Any finite subset of function values is jointly Gaussian
- Defined by a mean function and a kernel function
- Sampled "functions" are vectors plotted as curves

### 5. The Kernel (RBF)
- k(x, x') = exp(-0.5 * ((x-x')/ℓ)²)
- length_scale controls smoothness: small ℓ = wiggly, large ℓ = smooth
- The kernel replaces manually writing the covariance matrix

### 6. GP Regression — Toy Example
- 1D spectral interpolation: 3 observed bands → predict at new wavelengths
- Full pseudocode and numerical worked example (2 bands, 3 prediction points)
- LaTeX PDFs with every matrix multiplication shown
- Prior → posterior with uncertainty bands

### 7. Proofs Derived
- 2×2 matrix inverse derivation
- Posterior mean and covariance formulas (completing the square)
- Step 6 covariance computation with all intermediate numbers
- OLS closed-form solution w = (X^T X)^{-1} X^T y
- Matrix calculus rules (scalar derivatives → matrix notation)

### 8. OLS Linear Regression (Matrix Form)
- Design matrix with ones column for intercept
- Closed-form, deterministic, no gradient descent
- Full numerical example with 2 samples
- Connection to sklearn implementation

### 9. Bayesian Linear Regression (BLR)
- Prior over weights → likelihood → posterior (Bayes' theorem)
- Posterior mean and covariance formulas (B-8, B-9)
- Prediction with two sources of uncertainty (epistemic + aleatoric)
- OLS is a special case of BLR (infinitely vague prior)
- Sequential update visualization (0, 1, 3, 10 samples)
- Code from scratch on house price data

### 10. Bayes' Theorem Deep Dive
- Derived from conditional probability definition
- Soil carbon example with full rectangle visualization
- Prior, likelihood, evidence, posterior — all grounded in concrete numbers
- Why p(y) is constant w.r.t. w (integration sums out all w)
- Proportionality (∝) explained

### 11. OLS vs BLR vs GP Comparison
- Same house data, three methods side by side
- BLR = uncertainty over 2 parameters (forced linear)
- GP = uncertainty over the entire function (no shape assumed)
- GP is "0 explicit parameters, infinite implicit flexibility"

### 12. Multi-Feature GP
- Covariance matrix is always samples × samples (not features × features)
- Features are consumed by the kernel to produce similarity scores
- More features = higher-dimensional distance in kernel, same GP machinery
- Deep Kernel Learning = neural network + GP combination

## What Was NOT Covered (Remaining Items)

1. Multi-sample GP worked example (3 soil samples, 2 bands → SOC prediction)
2. Kernel hyperparameter optimization (maximum likelihood estimation)
3. Other kernel types (linear, Matérn, periodic, combined)
4. GP for classification
5. Scalability solutions (sparse GP, inducing points)
6. Linear algebra prerequisites (eigenvalues, eigenvectors, Cholesky, positive definiteness)

## Files in This Archive

- `code/` — All Python scripts (step01 through step06, OLS, BLR, GP comparison, Bayes visuals, LaTeX generators)
- `latex/` — LaTeX source files and build artifacts
- `docs/` — PDFs, PNGs, worked examples in text format, house_data.csv
