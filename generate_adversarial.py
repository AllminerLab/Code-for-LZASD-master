import pandas as pd
import json
from gensim.models import Word2Vec, KeyedVectors
import joblib
import numpy as np
from tqdm import tqdm
import os

train_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/train_set/train_data.txt',
                            sep='\x01', dtype=str)
test_data = pd.read_csv('/Users/lizhifm/code/python/导流广告/dataset/test_set/test_data.txt',
                        sep='\x01', dtype=str)
all_data = train_data.append(test_data).reset_index(drop=True)
w2v = KeyedVectors.load_word2vec_format('/Users/lizhifm/code/python/导流广告/w2v_content.w2v', binary=True, unicode_errors='ignore')
xgb = joblib.load('/Users/lizhifm/code/python/导流广告/xgb.dat')
pos_data = all_data[all_data['label']=='1'].reset_index(drop=True)
pos_data.drop(axis=1, columns=['label'], inplace=True)
with open('/Users/lizhifm/code/python/导流广告/replace_words.json') as f:
    sim_word = json.load(f)
adversarial_data = []
for i in tqdm(range(len(pos_data))):
    pos_data['content'][i] = pos_data['content'][i].replace(' ', '')
    adversarial_data.append(pos_data['content'][i])
    content = pos_data['content'][i].split('\x02')
    for j, c in enumerate(content):
        if c in sim_word:
            c_vecs = np.sum([att*w2v[w] for w, att in sim_word[c].items()], axis=0)
            fre_sum = sum([att for w, att in sim_word[c].items()])
        else:
            c_vecs = w2v[c]
            fre_sum = 1
        c_vecs = c_vecs / fre_sum
        score = xgb.predict_proba(c_vecs.reshape(1,100))[0][1]
        if score > 0.8 and c in sim_word:
            for sim in sim_word[c]:
                content[j] = sim
                adversarial_data.append('\x02'.join(content))
adversarial_path = "/Users/lizhifm/code/python/导流广告/dataset/adversarial_set/"
adversarial_data = pd.DataFrame({'content':adversarial_data})
os.makedirs(adversarial_path, exist_ok=True)
adversarial_data.to_csv(adversarial_path+'adversarial_data.txt', columns=['content'], sep='\x01', index=False)
