# -*- coding: UTF-8 -*-

from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# 将数据集根据feature与value的大小关系切分成两部分．其中大于value的为mat0，小于等于value的为mat1
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

# 生成叶节点．在回归树中，就是根据value的均值生成．
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

# 计算目标变量的方差的和
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

# 找到叶子节点的回归系数
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

# 返回dataSet的回归系数预测的值和实际值的误差的平方
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # tolS表示容许的误差下降值，tolN表示切分的最小样本数
    tolS = ops[0]; tolN = ops[1]
    # 如果数据集中的value都处于同一类了，那么就退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    # 计算数据集的方差和
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): # 迭代每个特征，找出最适合进行split的特征
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): # 找出此特征的全部的值并迭代
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) # 将此数据集按照此特征以及此特征值拆分成两个数据集
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue # 只要拆分后的数据集有一个小于切分的最小样本数，就放弃此次拆分
            newS = errType(mat0) + errType(mat1) # 计算拆分后的两个数据集的方差和
            if newS < bestS: # 如果方差和是最小的,就保存并继续迭代，看看能不能找到方差和更小的
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果拆分以后，误差已经小于阈值，那么就返回
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果拆分以后，每个子节点的数据都少于最小样本数，那么这个数据集就是一个叶子节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    # 如果不少于最小样本数，则进一步进行拆分
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

# leafType表示建立叶节点的函数
# errType表示误差计算函数
# ops包含树构建所需要的其它参数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops) # 找到最适合进行拆分的特征，已经拆分的特征值
    if feat == None: return val # 如果返回None，说明dataSet本身就是叶子节点了，或者说只有一个节点的树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 将数据集拆分成两部分，并分别建立左右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 判断obj是否是叶子节点
def isTree(obj):
    return (type(obj).__name__=='dict')

# 获取tree的平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    # 平均值是左右子树的平均值/2
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    # 如果没有testData,那么仅仅返回tree的平均值
    if shape(testData)[0] == 0: return getMean(tree)
    # 如果存在左右子树(不是叶子节点),则先对testData根据特征以及特征值进行拆分
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果存在左子树,那就对左子树通过testData的左子树进行剪枝
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    # 如果存在右子树,那就对右子树通过testData的右子树进行剪枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    # 如果没有左右子树,则确定是否可以进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算不合并的误差
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        # 计算合并的误差
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        # 如果合并后的误差小,那么进行合并
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else: return tree
    else: return tree

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat