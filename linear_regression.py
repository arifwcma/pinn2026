import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_data.csv")
x_raw = df["size_sqm"].values
y = df["price_lakh"].values
num_samples = len(x_raw)

ones_column = np.ones(num_samples)
X = np.column_stack([ones_column, x_raw])

XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
weights = XtX_inv @ Xty

fitted_intercept = weights[0]
fitted_slope = weights[1]

x_line = np.linspace(20, 130, 100)
y_line = fitted_intercept + fitted_slope * x_line

y_predicted = X @ weights
residuals = y - y_predicted
sum_squared_error = np.sum(residuals ** 2)

plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y, color="steelblue", s=60, zorder=5, label="Data")
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Fit: y = {fitted_intercept:.2f} + {fitted_slope:.2f}x")
plt.xlabel("House Size (sqm)")
plt.ylabel("Price (lakh)")
plt.title("Ordinary Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Intercept (w0): {fitted_intercept:.4f}")
print(f"Slope (w1):     {fitted_slope:.4f}")
print(f"Sum of squared errors: {sum_squared_error:.4f}")
print(f"\nTrue values were: intercept=1.0, slope=2.5")
