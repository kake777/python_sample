#数値微分の例
import numpy as np
from common_function import function_1
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1) #0から20まで0.1刻みのx配列
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
