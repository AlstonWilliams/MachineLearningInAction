# -*- coding: UTF-8 -*-

from numpy import *

def loadSimpleData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

# 单层决策树，即只根据一个特征进行分类
# 根据threshIneq：
#   - 如果是'lt',那么将dimen这个特征的值与threshVal对比，小于threshVal则将对应的结果设置为-1.否则则为+1
#   - 如果是'gt',则和上面相反
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 找到数据集上最佳的单层决策树
#   D表示数据的权重向量
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataArr)
    numSteps = 10.0; # 在特征的所有可能值上进行遍历
    bestStump = {}; # bestStump存储最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n): # 在数据集的所有特征上遍历
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max();
        # 为什么要很多步?一步不行么?
        stepSize = (rangeMax - rangeMin) / numSteps # 每一步中这个特征的取值范围
        for j in range(-1, int(numSteps) + 1): # 从每一步中这个特征的取值范围中寻找最佳单层决策树
            for inequal in ['lt', 'gt']: # 分别对比用这个特征正反类分类的错误率
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) # 用单层决策树预测结果
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0 # 如果单层决策树预测正确，则对应的行错误率是0
                weightedError = D.T * errArr # 根据错误率和错误权重计算带权重的错误率
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 寻找具有最低错误率的特征，特征值，和操作符(lt/gt)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m) # 每行的权重。初始为1/m。
    aggClassEst = mat(zeros((m,1))) # 每行的类别估计累计值
    for i in range(numIt):
        # 找到这次迭代的最佳单层决策树。包括最小错误率以及估计的类别向量
        # 会根据前面的分类器的错误率进行调整
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16))) # 根据错误率计算每个弱分类器的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst) # 如果样本被正确分类，则升高权重，如果被错误分类，则降低权重
        D = multiply(D, exp(expon))
        D = D/D.sum() # 更新没一行的权重
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0: break
    return weakClassArr

# 用每个分类器训练一个结果，然后用权值相加并通过sigmoid函数计算结果分类
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)