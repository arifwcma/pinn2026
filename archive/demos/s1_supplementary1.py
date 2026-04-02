import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
initial_shape = 2.0 / np.cosh(x)

plt.figure(figsize=(8, 4))
plt.plot(x, initial_shape, 'b-', linewidth=2)
plt.xlabel('Position x')
plt.ylabel('h(0, x)')
plt.title('Initial condition: h(0, x) = 2 sech(x)')
plt.axhline(y=0, color='gray', linewidth=0.5)
plt.tight_layout()
plt.savefig('demos/s1_supplementary1.png', dpi=150)
plt.show()
