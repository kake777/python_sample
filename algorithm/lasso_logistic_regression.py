import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
#pyton2.x
#from itertools import izip
#python3.x
import itertools
zip = getattr(itertools, 'izip', zip)


def get_data(data_fname, label_fname):
    result_data = []
    result_labels = []
    with open(data_fname, 'r') as f_data, open(label_fname, 'r') as f_label:
        for data, label in zip(f_data, f_label):
            result_data.append(data.rstrip().split(','))
            result_labels.append(int(label.rstrip()))
    return np.array(result_data, dtype=np.float64), result_labels

if __name__=='__main__':
    data_fname = 'datas/train800.csv'
    label_fname = 'datas/trainLabels800.csv'
    test_data_fname = 'datas/test200.csv'
    test_label_fname = 'datas/testLabels200.csv'

    data, labels = get_data(data_fname, label_fname)
    test_data, test_labels = get_data(test_data_fname, test_label_fname)

    lr = LogisticRegression()
    model = lr.fit(data, labels)
    y_pred = model.predict(test_data)
    print('accuracy:', model.score(test_data, test_labels))
    print('recall:', recall_score(test_labels, y_pred))
