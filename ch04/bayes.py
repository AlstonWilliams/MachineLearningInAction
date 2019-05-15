# -*- coding: UTF-8 -*-
from math import log
from numpy import *
from numpy.ma import array


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
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec

# trainCategory表示全部文档所属类别的向量.1表示是侮辱性文本,0则相反
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainDocs # 计算侮辱性文本在总文本中所占的比例,即p(c=1)
    p0Num = ones(numWords); p1Num = ones(numWords) # 初始化为1来避免概率中有0导致p(w|c)为0
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i] # 侮辱性文本中, 各个词汇的数量
            p1Denom += sum(trainMatrix[i]) # 侮辱性文本中总的词汇数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 取对数来防止下溢造成的影响
    p1Vect = log(p1Num / p1Denom) # 侮辱性文本中,每个词汇出现的概率,即p(wi|c=1), i为此词汇的索引
    p0Vect = log(p0Num / p0Denom) # 非侮辱性文本中,每个词汇出现的概率,即p(wi|c=0), i为此词汇的索引
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算log[p(w|c=i)]
    #   = log[p(w|c=i)] + log[p(c=i)]
    #   = log[p(w=0|c=i)p(w=1|c=i)....p(w=j|c=i)] + log[p(c=i)]
    #   = log[p(w=0|c=i)] + log[p(w=1|c=i)] + ... + log[p(w=j|c=i)] + log[p(c=i)]
    #   = sum(vec2Classify * piVec) + log(pClassi)

    # 由于p(w)在对于c=1或者c=0来说，都是个常数，所以可以忽略掉。
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50)
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText