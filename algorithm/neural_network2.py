import numpy as np

#シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#恒等関数(あくまでも例なので値を戻しているだけ)
def identity_function(x):
    return x

#0層から1層
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print('X.shape:', X.shape)
print('W1.shape:', W1.shape)
print('B1.shape:', B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print('A1:', A1)
print('Z1:', Z1)

#1層から2層
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print('Z1.shape:', Z1.shape)
print('W2.shape:', W2.shape)
print('B2.shape:', B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print('A2:', A2)
print('Z2:', Z2)

#2層から出力層
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print('Z2.shape:', Z2.shape)
print('W3.shape:', W3.shape)
print('B3.shape:', B3.shape)

A2 = np.dot(Z2, W3) + B3
Y = identity_function(A2)

print('A2:', A2)
print('Y:', Y)
