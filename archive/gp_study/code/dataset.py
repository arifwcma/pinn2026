import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

true_slope = 2.5
true_intercept = 1.0
noise_std = 1.5

house_size = np.array([30, 45, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
price = true_intercept + true_slope * house_size + np.random.normal(0, noise_std, size=len(house_size))
price = np.round(price, 1)

df = pd.DataFrame({"size_sqm": house_size, "price_lakh": price})
df.to_csv("house_data.csv", index=False)

plt.scatter(house_size, price, color="steelblue", s=60)
plt.xlabel("House Size (sqm)")
plt.ylabel("Price (lakh)")
plt.title("10 Houses: Size vs Price")
plt.tight_layout()
plt.show()

print(df.to_string(index=False))
