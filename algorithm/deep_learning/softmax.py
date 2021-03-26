import numpy as np

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

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print('softmax:', y)
print('sum:', np.sum(y))
