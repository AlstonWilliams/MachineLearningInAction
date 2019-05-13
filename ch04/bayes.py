# -*- coding: UTF-8 -*-

from numpy import *

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my vocabulary!" % word
    return returnVec

# trainCategory表示全部文档所属类别的向量.1表示是侮辱性文本,0则相反
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainDocs # 计算侮辱性文本在总文本中所占的比例,即p(c=1)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i] # 侮辱性文本中, 各个词汇的数量
            p1Denom += sum(trainMatrix[i]) # 侮辱性文本中总的词汇数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom # 侮辱性文本中,每个词汇所占的比例,即p(w|c=1)
    p0Vect = p0Num / p0Denom # 非侮辱性文本中,每个词汇所占的比例,即p(w|c=0)
    return p0Vect, p1Vect, pAbusive
