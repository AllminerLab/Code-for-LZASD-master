import pandas as pd
import jieba_fast as jieba
from tqdm import tqdm
import re
import os
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import word2vec, Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import joblib
from ssc_similarity import *
import json
np.random.seed(100)

class DataSet:
    def __init__(self, key_list=['content'], low_freq=0.001, high_freq=0.005, sim_thre=0.8):
        unlabeled_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/adversarial_set/adversarial_data.txt',
                                sep='\x01', dtype=str)
        print('#len, ', len(unlabeled_data))
        self.x_unlabel = {}
        print('generating unlabeled set')
        for key in key_list:
            self.x_unlabel[key] = []
            for i in tqdm(range(len(unlabeled_data))):
                try:   
                    content = unlabeled_data[key][i].split('\x02')
                except:
                    content = ['nan']
                if '' in content:
                    content.remove('')
                self.x_unlabel[key].append(content)
        with open('/Users/lizhifm/code/python/导流广告/replace_words.json') as f:
            self.sim_word = json.load(f)
        self.y_unlabel = np.ones(len(self.x_unlabel['content']))

    def getTrainTest(self):

        return self.sim_word, self.x_unlabel, self.y_unlabel

class W2V:
    def __init__(self, model_path, x_unlabel, sim_word, size=100, window=5, epoch_num=10):
        self.model = {}
        for key in x_unlabel:
            model = KeyedVectors.load_word2vec_format('/Users/lizhifm/code/python/导流广告/w2v_content.w2v', binary=True, unicode_errors='ignore')
            self.model[key] = model
            self.sim_word = sim_word

        self.x_unlabel = x_unlabel
        
    def getVecs(self, x):
        vecs_list = []
        for key in x:
            corpus = x[key]
            vecs = []
            for z in tqdm(corpus):
                z_vecs = []
                for c in z:
                    if c == '，' or c == '':
                        continue
                    if key == 'content' and c in self.sim_word:
                        c_vecs = np.sum([att*self.model[key][w] for w, att in self.sim_word[c].items()], axis=0)
                        fre_sum = sum([att for w, att in self.sim_word[c].items()])
                    else:
                        c_vecs = self.model[key][c]
                        fre_sum = 1
                    c_vecs = c_vecs / (fre_sum + 1e-7)
                    z_vecs.append(c_vecs)
                z_vecs = np.mean(z_vecs, axis=0)
                vecs.append(z_vecs)
            vecs = np.vstack(vecs)
            vecs_list.append(vecs)
        
        return np.hstack(vecs_list)
    
    def get_vectors(self):
        print('getting vectors')
        return self.getVecs(self.x_unlabel)

class Classifier:
    def __init__(self, model_path, max_depth=5, n_estimators=100):
        self.model = joblib.load(model_path)
        #self.model = LogisticRegression()
    
    def evaluate(self, model, threshold, vecs, y):
        pred_probas = model.predict_proba(vecs)[:,1]
        pred = np.zeros_like(pred_probas)
        pred[pred_probas > threshold] = 1
        precision = sum((pred!=0) & (y!=0))/sum(pred!=0)
        recall = sum((pred!=0) & (y!=0))/sum(y!=0)
        f1 = 2*(precision*recall/(precision+recall))
        print(f'Precision:{precision}')
        print(f'Recall:{recall}')
        print(f'F1-score:{f1}')
        return pred

    def train(self, unlabel_vecs, y_unlabel, threshold=0.6):

        print('Test Evaluation')
        unlabel_pred = self.evaluate(self.model, threshold, unlabel_vecs, y_unlabel)
        unlabel_prob = self.model.predict_proba(unlabel_vecs)[:,1]

        return unlabel_prob

    def ROC_curve(self, test_vecs, y_test):
        pred_probas = self.model.predict_proba(test_vecs)[:,1]

        fpr,tpr,_ = roc_curve(y_test, pred_probas)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.show()

import sys
import numpy as np
import gensim

key_list = ['content']
size = 100
dataset = DataSet(key_list=key_list, low_freq=0.001, high_freq=0.005, sim_thre=0.8)
sim_words, x_unlabel, y_unlabel = dataset.getTrainTest()
'''
d2v_model = D2V(x_train, x_test, size=size)
d2v_model.train(size=size, epoch_num=10)
train_vecs,test_vecs = d2v_model.get_vectors(size=size)
'''
w2v_model = W2V('/Users/lizhifm/code/python/导流广告/', x_unlabel, sim_words, size=size, window=5, epoch_num=5)
unlabel_vecs = w2v_model.get_vectors()
classifier_model = Classifier('/Users/lizhifm/code/python/导流广告/xgb.dat', max_depth=6, n_estimators=200)
unlabel_prob = classifier_model.train(unlabel_vecs, y_unlabel, threshold=0.7)

unlabeled_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/adversarial_set/adversarial_data.txt',
                                sep='\x01', dtype=str)

unlabeled_data['corpus'] = dataset.x_unlabel['content']
unlabeled_data['pred'] = unlabel_prob

unlabeled_data.to_csv('/Users/lizhifm/code/python/导流广告/adversarial_pred.txt', sep=',', index=False)

#classifier_model.ROC_curve(test_vecs, y_test)
