#coding:utf-8

import numpy as np
import pandas as pd
from gensim.models import word2vec
import logging

#读入数据
def load_data(path):
    data = np.load(path, allow_pickle=True)
    return data

x_train = load_data('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/x_train_data.npy')
x_test = load_data('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/x_test_data.npy')
y_train = load_data('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/y_train_data.npy')
y_test = load_data('D:/project_python/ClassLearning/NO4/Word2Vec sentiment/data/y_test_data.npy')
print(x_train[0])

# 打印日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#训练模型
#model = word2vec.Word2Vec(x_train,sg=0,hs=1,min_count=1,window=5,size=300)
def model(data,model_path):
    model = word2vec.Word2Vec(data,sg=0,hs=1,min_count=1,window=5,vector_size=300)
    model.save(model_path)
    return model
#训练数据集
train_model = model(x_train,'D:/project_python/ClassLearning/NO4/Word2Vec sentiment/train_model.model')
test_model = model(x_test,'D:/project_python/ClassLearning/NO4/Word2Vec sentiment/test_model.model')
