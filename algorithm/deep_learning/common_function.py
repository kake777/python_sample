import numpy as np

# ステップ関数の実装
def step_function(x):
    return np.array(x > 0, dtype=np.int32)

#シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

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

#2次関数(簡単な微分)
def function_1(x):
    return 0.01*x**2 + 0.1*x

#偏微分
def function_2(x):
    #return x[0]**2 + x[1]**2
    #または return np.sum(x**2)
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

#勾配
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x) #Xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val #値を元に戻す

    return grad

def numerical_gradient(f, x):
    #if X.ndim == 1:
    #    return _numerical_gradient_no_batch(f, X)
    #else:
    #    grad = np.zeros_like(X)
    #    
    #    for idx, x in enumerate(X):
    #        grad[idx] = _numerical_gradient_no_batch(f, x)
    #    
    #    return grad
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

#勾配降下法
# f : 最適化したい関数
# init_x : 初期値
# lr : learning rate
# step_num : 勾配法による繰り返しの数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


