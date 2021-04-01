#数値微分の例
import numpy as np
from common_function import function_1, numerical_diff
import matplotlib.pylab as plt

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x

    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1) #0から20まで0.1刻みのx配列
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
