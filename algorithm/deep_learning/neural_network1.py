import numpy as np
import matplotlib.pylab as plt

# ステップ関数の実装
def step_function(x):
    return np.array(x > 0, dtype=np.int32)

#シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU関数の実装
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
#y = step_function(x) #ステップ関数
#y = sigmoid(x) #シグモイド関数
y = relu(x) #ReLU関数
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y軸の範囲を指定
plt.show()
