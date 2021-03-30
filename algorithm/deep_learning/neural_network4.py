#MNISTデータのロード処理
from dataset.mnist import load_mnist
import numpy as np
import pickle

#シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ソフトマックス関数
#※ソフトマックス関数はニューラルネットワークの学習時に使用するものであり、
#  推論(分類)時には省略するのが一般的
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("algorithm/deep_learning/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0 #認識精度
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) #最も確率の高い要素のインックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
