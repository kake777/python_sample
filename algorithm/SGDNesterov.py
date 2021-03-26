# coding: utf-8
import numpy as np
import math
from sklearn.metrics import accuracy_score, recall_score
#pyton2.x
#from itertools import izip
#python3.x
import itertools
zip = getattr(itertools, 'izip', zip)


class SgdNesterov:
    def __init__(self, feat_dim, loss_type='log', mu=0.9, learning_rate=0.5):
        self.weight = np.zeros(feat_dim)  # features weight
        self.loss_type = loss_type  # type of loss function
        self.feat_dim = feat_dim
        self.x = np.zeros(feat_dim)
        self.mu = mu  # momentum
        self.t = 1  # update times
        self.v = np.zeros(feat_dim)
        self.learning_rate = learning_rate

    def fit(self, data_fname, label_fname):
        with open(data_fname, 'r') as f_data, open(label_fname, 'r') as f_label:
            for data, label in zip(f_data, f_label):
                self.features = np.array(data.rstrip().split(','), dtype=np.float64)
                y = int(-1) if int(label.rstrip())<=0 else int(1)  # posi=1, nega=-1に統一する
                # update weight
                self.update(y)
                self.t += 1
        return self.weight

    def predict(self, features): #margin
        return np.dot(self.weight, features)

    def calc_loss(self,m): # m=py=wxy
        if self.loss_type == 'hinge':
            return max(0,1-m)
        elif self.loss_type == 'log':
            if m<=-700: m=-700
            return math.log(1+math.exp(-m))

    # gradient of loss function
    def calc_dloss(self,m): # m=py=wxy
        if self.loss_type == 'hinge':
            res = -1.0 if (1-m)>0 else 0.0 # lossが0を超えていなければloss=0.そうでなければ-mの微分で-1になる
            return res
        elif self.loss_type == 'log':
            if m < 0.0:
                return float(-1.0) / (math.exp(m) + 1.0) # yx-e^(-m)/(1+e^(-m))*yx
            else:
                ez = float( math.exp(-m) )
                return -ez / (ez + 1.0) # -yx+1/(1+e^(-m))*yx

    def update(self, y):
        w_ahead = self.weight + self.mu * self.v
        pred = np.dot(w_ahead, self.features)
        grad = y*self.calc_dloss(y*pred)*self.features # gradient
        self.v = self.mu * self.v - self.learning_rate * grad # velocity update stays the same
        # update weight
        self.weight += self.v


if __name__=='__main__':
    data_fname = 'datas/train800.csv'
    label_fname = 'datas/trainLabels800.csv'
    test_data_fname = 'datas/test200.csv'
    test_label_fname = 'datas/testLabels200.csv'

    sgd_n = SgdNesterov(40, loss_type='hinge')
    sgd_n.fit(data_fname, label_fname)
    y_true = []
    y_pred = []
    with open(test_data_fname, 'r') as f_data, open(test_label_fname, 'r') as f_label:
        for data, label in zip(f_data, f_label):
            pred_label = sgd_n.predict(np.array(data.rstrip().split(','), dtype=np.float64))
            y_true.append(int(label))
            y_pred.append( 1 if pred_label>0 else 0)
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred))
