from utils import *
import sys
import numpy as np
import gensim

key_list = ['content']
size = 100
dataset = DataSet(key_list=key_list, low_freq=0.001, high_freq=0.005, sim_thre=0.8)
x_train, x_test, y_train, y_test, sim_words = dataset.getTrainTest()
'''
d2v_model = D2V(x_train, x_test, size=size)
d2v_model.train(size=size, epoch_num=10)
train_vecs,test_vecs = d2v_model.get_vectors(size=size)
'''
w2v_model = W2V(x_train, x_test, sim_words, size=size, window=5, epoch_num=5)
train_vecs, test_vecs = w2v_model.get_vectors()
classifier_model = Classifier(max_depth=6, n_estimators=200)
train_pred, test_pred = classifier_model.train(train_vecs, y_train, test_vecs, y_test, threshold=0.7)
train_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/train_set/train_data.txt',
                                sep='\x01', dtype=str)
test_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/test_set/test_data.txt',
                                sep='\x01', dtype=str)

train_data['corpus'] = dataset.x_train['content']
train_data['pred'] = train_pred
test_data['corpus'] = dataset.x_test['content']
test_data['pred'] = test_pred

false_train = train_data[dataset.y_train!=train_pred]
false_test = test_data[dataset.y_test!=test_pred]
train_data.to_csv('/Users/lizhifm/code/python/导流广告/train_pred.txt', sep=',', index=False)
test_data.to_csv('/Users/lizhifm/code/python/导流广告/test_pred.txt', sep=',', index=False)
false_train.to_csv('/Users/lizhifm/code/python/导流广告/false_train.txt', sep=',', index=False)
false_test.to_csv('/Users/lizhifm/code/python/导流广告/false_test.txt', sep=',', index=False)
#classifier_model.ROC_curve(test_vecs, y_test)
