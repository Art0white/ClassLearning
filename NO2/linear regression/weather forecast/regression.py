# https://tianqi.2345.com/wea_forty/71654.htm

from bs4 import BeautifulSoup
import re
import requests
import numpy as np
import matplotlib.pyplot as plt
import random
import regression1
import regression3
import regression4

#爬取网页
def crawb(retX,retY):
    url = 'https://www.tianqishi.com/lishi/beijing.html'
    page = requests.get(url)
    page.encoding = 'utf-8'
    page = page.text
    pagesoup = BeautifulSoup(page, "html.parser")
    weathertable = pagesoup.find_all(name= "table", attrs={"class":"yuBaoTable"})

    # 爬取数据
    for link in weathertable[0].find_all("tr"):
        # 爬取最小温度与最大温度
        temp = link.find_all("td")[1]
        temps = temp.text.split('~')
        temps[1] = temps[1].replace("℃", "")
        # print(temps[0] + " and " + temps[1])

        # 爬取时间
        daytd = link.find_all("td")[0]
        day = daytd.find_all("a")[0].text

        # 爬取风向
        nFeng = 0; sFeng = 0; wFeng = 0; eFeng = 0
        wind = link.find_all("td")[3].text
        if "北" in wind:
            nFeng = 1
        if "南" in wind:
            sFeng = 1
        if "西" in wind:
            wFeng = 1
        if "东" in wind:
            eFeng = 1

        # 爬取风力
        windpowerall = link.find_all("td")[4].text
        windpowers = windpowerall.split('-')

        # 爬取天气情况
        # 晴10 > 多云9 > 扬沙8 > 阴7 > 小雨6 > 中雨5 > 大雨4 > 小雪3 > 中雪2 > 大雪1
        weatherval = 0
        weather = link.find_all("td")[2].text
        if "大雪" in weather:
            weatherval = 1
        elif "中雪" in weather:
            weatherval = 2
        elif "小雪" in weather:
            weatherval = 3
        elif "大雨" in weather:
            weatherval = 4
        elif "中雨" in weather:
            weatherval = 5
        elif "小雨" in weather:
            weatherval = 6
        elif "阴" in weather:
            weatherval = 7
        elif "扬沙" in weather:
            weatherval = 8
        elif "多云" in weather:
            weatherval = 9
        elif "晴" in weather:
            weatherval = 10

        # 自变量：最低温度；最高温度；时间；北风；南风；西风；东风；风力
        retX.append([int(temps[0]), int(temps[1]), int(day), nFeng, sFeng, wFeng, eFeng, int(windpowers[1])])
        # 因变量：天气情况
        retY.append(weatherval)

def crossValidation(xArr,yArr,numVal=10):#交叉验证岭回归，numVal为交叉验证次数
    m=len(yArr)
    indexList=list(range(m))
    errorMat=np.zeros((numVal,30))
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList)
        for j in range(m):#随机生成测试集和训练集
            if j <m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=regression4.ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX=np.mat(testX);matTrainX=np.mat(trainX)
            meanTrain=np.mean(matTrainX,0)
            varTrain=np.var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            yEst=matTestX*np.mat(wMat[k,:]).T+np.mean(trainY)
            errorMat[i,k]=regression3.rssError(yEst.T.A,np.array(testY))
    meanErrors=np.mean(errorMat,0)
    minMean=float(min(meanErrors))
    bestWeights=wMat[np.nonzero(meanErrors==minMean)]
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    meanX=np.mean(xMat,0);varX=np.var(xMat,0)
    unReg=bestWeights/varX
    print('%f%+f*最低温度%+f*最高温度%+f*时间%+f*北风%+f*南风%+f*西风%+f*东风%+f*风力'%((-1*np.sum(np.multiply(meanX,unReg))+np.mean(yMat)),unReg[0,0],unReg[0,1],unReg[0,2],unReg[0,3],unReg[0,4],unReg[0,5],unReg[0,6],unReg[0,7]))
    print(4672.123024-0.360756*9 + 0.324222*24 - 0.000231*20220513 - 0.083877*1 + 0.242398*0 + 0.224642*1 - 0.868138*0 - 0.355710*4)

if __name__=="__main__":
    lgX=[];lgY=[]
    crawb(lgX, lgY)
    crossValidation(lgX, lgY)  # 交叉验证岭回归
