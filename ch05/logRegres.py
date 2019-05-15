# -*- coding: UTF-8 -*-
import random

from numpy import mat, shape, ones
from numpy.ma import exp, array, arange


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# 计算sigmoid函数
# 为什么使用这个函数?因为它有阶跃函数的性质，x大于0的时候，结果趋向与1;小于0时，结果趋向-1
# 也就是说，它适合二类分类问题，但是对于多类分类问题不行。
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() # mat(classLabels)得到的是一个一行m列的矩阵，transpose()将它转换成m行一列的矩阵. m为训练数据集的行数
    m, n = shape(dataMatrix) # 行数和数据的维度数
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1)) # 每个维度的权重
    for k in range(maxCycles): # 需要计算很多次.次数太少达不到效果，太多会不会造成过拟合?
        h = sigmoid(dataMatrix * weights) # 对dataMatrix和每个维度的权重相乘求一个label
        error = (labelMat - h) # 将上述的结果和实际的label的差异当作error
        weights = weights + alpha * dataMatrix.transpose() * error # 根据上面的error和alpha改进权重
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 跟gradAscent相比，前者需要走很多遍数据集，才能得到结果。而随即梯度下降只需要走一遍就好。
# 但是只运行一次的效果不如gradAscent的好
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 相对于stocGradAscent0，这个方法不仅迭代了更多次提高精度，而且每次迭代时通过随机选取样本的方式来更新回归系数
# 每次迭代还是会读取全部的数据集
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01 # 保证alpha永远不等于0，让它不至于在多次迭代后失去效果
            randomIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex] * weights))
            error = classLabels[randomIndex] - h
            weights = weights + alpha * error * dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))
