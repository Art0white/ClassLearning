def text():
    f1=open('good.txt','r',encoding='utf-8')
    f2=open('bad.txt','r',encoding='utf-8')
    line1=f1.readline()
    line2=f2.readline()
    str=''
    while line1:
        str+=line1
        line1=f1.readline()
    while line2:
        str+=line2
        line2=f2.readline()
    f1.close()
    f2.close()
    return str
#单个字作为特征
def bag_of_words(words):
    return dict([(word,True) for word in words])
#print(bag_of_words(text())

#把词语（双字）作为搭配，并通过卡方统计，选取排名前1000的词语
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用卡方统计的方法，选择排名前1000的词语
    newBigrams = [u + v for (u, v) in bigrams]
    #return bag_of_words(newBigrams)
#print(bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=1000))

# 把单个字和词语一起作为特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    newBigrams = [u + v for (u, v) in bigrams]
    a = bag_of_words(words)
    b = bag_of_words(newBigrams)
    a.update(b)  # 把字典b合并到字典a中
    return a  # 所有单个字和双个字一起作为特征
#print(bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=1000))

import jieba
# 返回分词列表如：[['我','爱','北京','天安门'],['你','好'],['hello']]，一条评论一个
def readfile(filename):
    stop=[line.strip() for line in open('stop.txt','r',
                                        encoding='utf-8').readlines()]
    f=open(filename,'r',encoding='utf-8')
    line=f.readline()
    str=[]
    while line:
        s=line.split('\t')
        fenci=jieba.cut(s[0],cut_all=False)
        str.append(list(set(fenci)-set(stop)-set(['\ufeff','\n'])))
        line=f.readline()
    return str

from nltk.probability import FreqDist,ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
# 获取信息量较高(前number个)的特征(卡方统计)
def jieba_feature(number):
    posWords=[]
    negWords=[]
    for items in readfile('good.txt'):
        for item in items:
            posWords.append(item)
    for items in readfile('bad.txt'):
        for item in items:
            negWords.append(item)
    word_fd=FreqDist()   # 可统计所有词的词频
    con_word_fd=ConditionalFreqDist()    # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd[word]+=1
        con_word_fd['pos'][word]+=1
    for word in negWords:
        word_fd[word]+=1
        con_word_fd['neg'][word]+=1
    pos_word_count=con_word_fd['pos'].N()    # 积极词的数量
    neg_word_count=con_word_fd['neg'].N()    # 消极词的数量
    # 一个词的信息量等于积极卡方统计量加上消极卡方统计量
    total_word_count=pos_word_count+neg_word_count
    word_scores={}
    for word,freq in word_fd.items():
        pos_score=BigramAssocMeasures.chi_sq(con_word_fd['pos'][word],(freq,
                                                                       pos_word_count),total_word_count)
        neg_score=BigramAssocMeasures.chi_sq(con_word_fd['neg'][word],(freq,
                                                                       neg_word_count),total_word_count)
        word_scores[word]=pos_score+neg_score
        best_vals=sorted(word_scores.items(),key=lambda item: item[1],
                         reverse=True)[:number]
        best_words=set([w for w,s in best_vals])
    return dict([(word,True) for word in best_words])
# 构建训练需要的数据格式：
# [[{'买': 'True', '京东': 'True', '物流': 'True', '包装': 'True', '\n': 'True', '很快': 'True', '不错': 'True', '酒': 'True', '正品': 'True', '感觉': 'True'},  'pos'],
# [{'买': 'True', '\n':  'True', '葡萄酒': 'True', '活动': 'True', '澳洲': 'True'}, 'pos'],
# [{'\n': 'True', '价格': 'True'}, 'pos']]
def build_features():
    #feature = bag_of_words(text())
    #feature = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=900)
    # feature =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=900)
    feature = jieba_feature(1000)  # 结巴分词
    posFeatures = []
    for items in readfile('good.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        posWords = [a, 'pos']  # 为积极文本赋予"pos"
        posFeatures.append(posWords)
    negFeatures = []
    for items in readfile('bad.txt'):
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        negWords = [a, 'neg']  # 为消极文本赋予"neg"
        negFeatures.append(negWords)
    return posFeatures, negFeatures
