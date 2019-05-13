# -*- coding: UTF-8 -*-

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # dataSet的尺寸
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 先将inX扩充到dataSet的行数,然后跟dataSet相减得到inX和dataSet中每一行的每个元素的差值
    sqDiffMat = diffMat ** 2 # 对每一行每个元素的差值求平方
    sqDistances = sqDiffMat.sum(axis=1) # 对每一行求差值的平方的和
    distances = sqDistances ** 0.5 # 对上面的结果取开方
    sortedDistIndices = distances.argsort() # 获取数组中距离从小到大的索引
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]] # 依次取k个最小的距离值对应的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 将此label包含的元素值+1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True) # 按照label包含的元素个数进行排序
    return sortedClassCount[0][0] # 返回包含元素最多的那个label

def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, "smallDoses": 2, 'didntLike': 3}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3] # 只修改当前index的数据.其中":"指的是全部列
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) # 获取每一列的最小值
    maxVals = dataSet.max(0) # 获取每一列的最大值
    ranges = maxVals - minVals # 获取每一列的范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1)) # 特征值相除
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # 1. 将原始数据转换成矩阵形式
    normMat, ranges, minVals = autoNorm(datingDataMat) # 2. normalization
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3) # 根据K均值算法计算分类
    print("You will probably like this person: ", resultList[classifierResult - 1])


def img2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifer came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
