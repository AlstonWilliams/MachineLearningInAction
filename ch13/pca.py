# -*- coding: UTF-8 -*-

from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

# topNfeat表示想要应用的特征的数量
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals # 去平均值
    covMat = cov(meanRemoved, rowvar=0) # 求协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat)) # 求特征值和特征向量
    eigValInd = argsort(eigVals)            # 由小到大排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 只取topNfeat个最高的特征值
    redEigVects = eigVects[:,eigValInd]       # 重新组织特征向量
    lowDDataMat = meanRemoved * redEigVects # 把数据转换到新的维度
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
