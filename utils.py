import pandas as pd
import jieba_fast as jieba
from tqdm import tqdm
import re
import os
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import word2vec
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
        train_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/train_set/train_data.txt',
                                 sep='\x01', dtype=str)
        test_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/test_set/test_data.txt',
                                sep='\x01', dtype=str)

        self.key = key_list
        self.x_train = {}
        self.x_test = {}
        self.x_unlabel = {}
        self.counts = {}
        self.hanzi_index = {}
        self.sim_word = {}
        self.replace_words = []
        print('generating train set')
        for key in self.key:
            self.x_train[key] = []
            for i in range(len(train_data)):
                try:
                    content = train_data[key][i].split('\x02')
                except:
                    content = ['nan']
                if '' in content:
                    content.remove('')
                if key == 'content':
                    for word in content:
                        self.counts[word] = self.counts.get(word, 0) + 1
                self.x_train[key].append(content)
        print('generating test set')
        for key in self.key:
            self.x_test[key] = []
            for i in range(len(test_data)):
                try:   
                    content = test_data[key][i].split('\x02')
                except:
                    content = ['nan']
                if '' in content:
                    content.remove('')
                if key == 'content':
                    for word in content:
                        self.counts[word] = self.counts.get(word, 0) + 1
                self.x_test[key].append(content)

        count_sum = 0
        for word in self.counts:
            count_sum += self.counts[word]
        for word in self.counts:
            self.counts[word] /= count_sum

        train_data.loc[:,'label'][train_data['label'] != '0'] = 1
        test_data.loc[:,'label'][test_data['label'] != '0'] = 1
        train_data.loc[:,'label'][train_data['label'] == '0'] = 0
        test_data.loc[:,'label'][test_data['label'] == '0'] = 0

        self.y_train = train_data['label'].values.astype('int')
        self.y_test = test_data['label'].values.astype('int')
        self.ssc_table = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/hanzi_ssc_res.txt', sep='\t',names=['Ucode','hanzi','ssc'])
        self.ssc_table = self.ssc_table.drop_duplicates(['hanzi'],keep='first').reset_index(drop=True)
        if 'sim_mat.pkl' in os.listdir('/Users/lizhifm/code/python/导流广告/dataset/'):
            with open('/Users/lizhifm/code/python/导流广告/dataset/sim_mat.pkl', 'rb') as f:
                self.sim_mat = joblib.load(f)
            with open('/Users/lizhifm/code/python/导流广告/dataset/hanzi_index.pkl', 'rb') as f:
                self.hanzi_index = joblib.load(f)
            with open('/Users/lizhifm/code/python/导流广告/dataset/m2t.pkl', 'rb') as f:
                self.m2t = joblib.load(f)
            
        else:
            self.hanzi_index = {}
            self.m2t = {}
            exits_count = 0
            for i, word in enumerate(self.ssc_table['hanzi']):
                if word in self.counts:
                    self.hanzi_index[word] = exits_count
                    self.m2t[exits_count] = i
                    exits_count += 1

            print(f'generating similarity matrix {exits_count}x{exits_count}')
            self.sim_mat = np.zeros([len(self.hanzi_index), len(self.hanzi_index)])

            for word_i in tqdm(self.hanzi_index):
                ind_i = self.hanzi_index[word_i]
                for word_j in self.hanzi_index:
                    ind_j = self.hanzi_index[word_j]
                    self.sim_mat[ind_i][ind_j] = computeSSCSimilaruty(self.ssc_table['ssc'][self.m2t[ind_i]],self.ssc_table['ssc'][self.m2t[ind_j]], 'ALL')

            with open('/Users/lizhifm/code/python/导流广告/dataset/sim_mat.pkl', 'wb') as f:
                joblib.dump(self.sim_mat, f)
            with open('/Users/lizhifm/code/python/导流广告/dataset/hanzi_index.pkl', 'wb') as f:
                joblib.dump(self.hanzi_index, f)
            with open('/Users/lizhifm/code/python/导流广告/dataset/m2t.pkl', 'wb') as f:
                joblib.dump(self.m2t, f)
            
        def replace_strategy(word):
            if word not in self.hanzi_index or word in self.sim_word:
                return
            candidate = np.where(self.sim_mat[self.hanzi_index[word]]>=sim_thre)[0]
            candidate = candidate.tolist()
            candidate.sort(key=lambda x: self.counts[self.ssc_table['hanzi'][self.m2t[x]]], reverse=True)
            sim_words = {}
            i = 0
            for ind in candidate:
                if i == 10:
                    break
                count = self.counts[self.ssc_table['hanzi'][self.m2t[ind]]]
                sim_words[self.ssc_table['hanzi'][self.m2t[ind]]] = count
                i += 1
            self.sim_word[word] = sim_words            
        
        def update(self, corpus):
            pass

        print('replacing low frequency words')
        for i in range(len(self.x_train['content'])):
            for word in self.x_train['content'][i]:
                replace_strategy(word)
        for i in range(len(self.x_test['content'])): # ['加','我'] -> {'加','伽',...,}'我','硪']
            for word in self.x_test['content'][i]:
                replace_strategy(word)
        print('薇', self.sim_word['薇'])
        print('徽', self.sim_word['徽'])
        print('威', self.sim_word['威'])

        with open('/Users/lizhifm/code/python/导流广告/replace_words.json', 'w+') as f:
            json.dump(self.sim_word, f, ensure_ascii=False)
    def getTrainTest(self):

        return self.x_train, self.x_test, self.y_train, self.y_test, self.sim_word

class W2V:
    def __init__(self, x_train, x_test, sim_word, size=100, window=5, epoch_num=10):
        self.model = {}
        for key in x_train:
            model = word2vec.Word2Vec(size=size, hs=1, min_count=1, window=window, alpha=0.05)
            model.build_vocab(x_train[key]+x_test[key])
            model.train(x_train[key], total_examples=model.corpus_count, epochs=epoch_num)
            model.wv.save_word2vec_format('/Users/lizhifm/code/python/导流广告/w2v_'+key+'.w2v', binary=True)
            self.model[key] = model
            self.sim_word = sim_word

        self.x_train = x_train
        self.x_test = x_test
        
    def getVecs(self, x):
        vecs_list = []
        for key in x:
            corpus = x[key]
            vecs = []
            for z in tqdm(corpus):
                z_vecs = []
                for c in z:
                    '''
                    if key == 'content' and c in self.sim_word:
                        c_vecs = np.sum([att*self.model[key][w] for w, att in self.sim_word[c].items()], axis=0)
                        fre_sum = sum([att for w, att in self.sim_word[c].items()])
                    else:
                        c_vecs = self.model[key][c]
                        fre_sum = 1
                    '''
                    c_vecs = self.model[key][c]
                    fre_sum = 1
                    c_vecs = c_vecs / fre_sum
                    z_vecs.append(c_vecs)
                z_vecs = np.mean(z_vecs, axis=0)
                vecs.append(z_vecs)
            vecs = np.vstack(vecs)
            vecs_list.append(vecs)
        
        return np.hstack(vecs_list)
    
    def get_vectors(self):
        print('getting vectors')
        return self.getVecs(self.x_train), self.getVecs(self.x_test)

class Classifier:
    def __init__(self, max_depth=5, n_estimators=100):
        #self.model = XGBClassifier(max_depth=max_depth, learning_rate=0.1, n_estimators=n_estimators, objective='reg:logistic',use_label_encoder=False, seed=100)
        self.model = LogisticRegression()
    
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

    def train(self, train_vecs, y_train, test_vecs, y_test, threshold=0.6):
        print('training classifier....')
        #self.model.train(train_vecs, y_train)
        self.model.fit(train_vecs, y_train)
        joblib.dump(self.model, '/Users/lizhifm/code/python/导流广告/xgb.dat')
        #print('Test Accuracy: %.2f'% self.model.score(test_vecs, y_test))
        print('Train Evaluation')
        train_pred = self.evaluate(self.model, threshold, train_vecs, y_train)
        print('Test Evaluation')
        test_pred = self.evaluate(self.model, threshold, test_vecs, y_test)

        return train_pred, test_pred

    def ROC_curve(self, test_vecs, y_test):
        pred_probas = self.model.predict_proba(test_vecs)[:,1]

        fpr,tpr,_ = roc_curve(y_test, pred_probas)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.show()





