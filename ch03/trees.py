# -*- coding: UTF-8 -*-
from math import log
import operator

# 香农信息熵是针对整个数据集的，而不是仅仅针对于某个特征
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 根据axis这一列划分新的数据集，新的数据集中只包含dataSet中axis列的值等于value的数据，并且不包含axis列
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # 获取i这个feature包含的值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算按照i这个feature进行分类的信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算按照i这个feature进行分类的信息增益相对于整个dataSet的信息熵的差值
        infoGain = baseEntropy - newEntropy
        # 拿到信息增益最大的feature
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount

# labels是各个feature的定义
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果dataSet中的全部数据都属于同一类别，那么返回这个类别，当作叶子节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果dataSet中仅有一列，即label列，那么返回占多数的label值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最适合进行拆分的feature
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    # 拿到这个feature下的不重复的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 递归调用，创建决策树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree