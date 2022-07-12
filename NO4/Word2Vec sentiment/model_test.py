#coding:utf-8

import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.svm import SVC
import joblib
from model import get_sent_vec
import jieba

#下载数据
train_vec = np.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/train_vec.npy')
test_vec = np.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/test_vec.npy')
y_train =np.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/y_train_data.npy')
y_test = np.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/y_test_data.npy')

#对单个句子进行情感判断
def svm_predict(sent):
    model = word2vec.Word2Vec.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/train_model.model')
    sent_cut = jieba.lcut(sent)
    sent_cut_vec = get_sent_vec(300,sent_cut,model)
    clf = joblib.load('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/svm_model/svm_model.pkl')
    result = clf.predict(sent_cut_vec)

    if int(result[0] == 1):
        print(sent,'pos')
    else:
        print(sent,'neg')


#情感正式开始预测
sent = '破手机，垃圾'
svm_predict(sent)