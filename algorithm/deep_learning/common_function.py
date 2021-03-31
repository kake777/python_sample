import numpy as np

# ステップ関数の実装
def step_function(x):
    return np.array(x > 0, dtype=np.int32)

#シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU関数の実装
def relu(x):
    return np.maximum(0, x)

#恒等関数(あくまでも例なので値を戻しているだけ)
def identity_function(x):
    return x

#この書き方ではオーバーフローの問題が出てくるので、このままでは利用できない
#def softmax(a):
#    exp_a = np.exp(a) #指数関数
#    sum_exp_a = np.sum(exp_a) #指数関数の和
#    y = exp_a / sum_exp_a
#
#    return y

#ソフトマックス関数
#※ソフトマックス関数はニューラルネットワークの学習時に使用するものであり、
#  推論(分類)時には省略するのが一般的
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

#2乗和誤差
def sum_squareed_error(y, t):
    return 0.5 * np.sum((y - t)**2)

#交差エントロピー誤差(バッチ対応版)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#数値微分
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

