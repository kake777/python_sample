import numpy as np
from common_function import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で初期化

    # 予測メソッド
    def predict(self, x):
        return np.dot(x, self.W)

    # 損失関数の値を求めるメソッド
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W) # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p)) # 最大値のインデックス

t = np.array([0, 0, 1]) # 正解ラベル
print(net.loss(x, t))

#def f(W):
#    return net.loss(x, t)
f = lambda w: net.loss(x, t)

dw = numerical_gradient(f, net.W) # 勾配
print(dw)
