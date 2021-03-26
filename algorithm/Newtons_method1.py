import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**3 - 2*x**2 + x + 3
d1f = lambda x: 3*x**2 - 4*x + 1
d2f = lambda x: 6*x - 4
f2 = lambda bar_x, x: f(bar_x) + d1f(bar_x)*(x - bar_x) + d2f(bar_x)*(x - bar_x)**2

x = np.linspace(-10, 10, num=100)
plt.plot(x, f(x), label="original")
plt.plot(x, f2(2, x), label="2nd order approximation")
plt.legend()
plt.xlim((-10, 10))
plt.show()
