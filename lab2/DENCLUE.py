from numpy import *
import numpy as np
import math
from collections import Counter
from ast import literal_eval

#求高斯核
def calculateGaussianK(Z):
    GaussianK = (1/pow(2*math.pi,len(Z)/2))*np.exp(-np.dot(Z,Z)/2)
    return GaussianK

#求概率密度
def densityEstimation(X,D,h):
    a = 0
    for i in range(len(D)):
        a += calculateGaussianK((X - D[i]) / h)
    densityEst = (1/(len(D)*pow(h,4)))*a
    return densityEst

#找密度吸引子
def findAttractor(x,D,h,p):
    Xt = x
    a = np.zeros((4))
    b = 0
    normCollct = [] #存放所有的Xt和Xtt的二范数
    for i in range(len(D)):
        a += calculateGaussianK((Xt - D[i])/h)*D[i]
        b += calculateGaussianK((Xt - D[i])/h)
    Xtt = a/b
    normCollct.append(np.linalg.norm(Xt - Xtt, axis=0))
    while(np.linalg.norm(Xt - Xtt, axis=0, keepdims=True) > p):
        a = np.zeros((4))
        b = 0
        Xt = Xtt
        for i in range(len(D)):
            a += calculateGaussianK((Xt - D[i])/h)*D[i]
            b += calculateGaussianK((Xt - D[i])/h)
        Xtt = a/b
        normCollct.append(np.linalg.norm(Xt - Xtt, axis=0))
    if(len(normCollct)<2):
        radius = normCollct[-1]
    else:
        radius = normCollct[-1] + normCollct[-2] # 取数组最后两个值的和为密度吸引对应的半径
    return Xt,radius

def densityReachable(densityAttractorCollection,attractorRadiusDict):
    attrCluster = []    #存放密度吸引子的二维数组
    while(densityAttractorCollection):
        temp_attrCluster = []
        temp_attrCluster.append(densityAttractorCollection[0])
        densityAttractorCollection.remove(densityAttractorCollection[0])
        for m in range(len(densityAttractorCollection)):
            for n in range(len(temp_attrCluster)):
                if (np.linalg.norm(np.array(literal_eval(temp_attrCluster[n])) - np.array(literal_eval(densityAttractorCollection[m])), axis=0)
                        <= attractorRadiusDict[temp_attrCluster[n]] + attractorRadiusDict[densityAttractorCollection[m]]):
                    temp_attrCluster.append(densityAttractorCollection[m])
                    break

        #这里从1开始，因为之前第52行的时候已经remove了temp_attrCluster的第一个数
        for p in range(1,len(temp_attrCluster)):
            densityAttractorCollection.remove(temp_attrCluster[p])
        attrCluster.append(temp_attrCluster)
    return attrCluster

    # for i in range(len(densityAttractorCollection)-1):
    #     temp_attrCluster = []  #暂时存放密度吸引子的数组
    #     temp_attrCluster.append(densityAttractorCollection[i])  #依次取一个密度吸引子到暂时的密度吸引子数组中
    #     for j in range(i+1,len(densityAttractorCollection)):     #再从i+1依次取密度吸引子进行距离比较
    #         for k in range(len(temp_attrCluster)):             #同时与放入暂时的密度吸引子数组中的密度吸引子进行距离比较
    #             #如果以密度吸引子为核心，以密度吸引子对应的半径为半径的两个圆相交，即可合并这两个吸引子及其吸引的所有点组成一个大簇
    #             if (np.linalg.norm(np.array(literal_eval(temp_attrCluster[k])) - np.array(literal_eval(densityAttractorCollection[j])), axis=0)
    #                     <= attractorRadiusDict[temp_attrCluster[k]] + attractorRadiusDict[densityAttractorCollection[j]]):
    #                 temp_attrCluster.append(densityAttractorCollection[j])
    #                 #只要发现与其中的一个吸引子可以合并，根据传递性就可以合并这两个簇了
    #                 break
    #     print(temp_attrCluster)
    #     #可能存在密度吸引子为[1,2,3,5]和[3,5]，这里要舍弃[3,5]，所以要判断暂时的密度吸引子数组是否为密度吸引子的二维数组中的子集
    #
    #     #一开始为空的时候要判断一下，不然永远不会for循环
    #     if (len(attrCluster) == 0):
    #         attrCluster.append(temp_attrCluster)
    #         continue
    #
    #     for m in range(len(attrCluster)):
    #         if(set(temp_attrCluster) <= set(attrCluster[m])):
    #             pass
    #         else:
    #             attrCluster.append(temp_attrCluster)
    #print("****************************")
    #print(attrCluster)
    #print(len(attrCluster))


def denclue(D,label,h,minDensity,p):
    densityAttractorCollection = [] #密度吸引子集合
    attractorRadiusDict = {}        #密度吸引子与其对应的半径的映射
    attractorPointsDict = {}        #密度吸引子与其吸引的所有点的映射
    attractorPointsLabelDict = {}   #密度吸引子与其吸引的点的种类标签的映射
    for i in range(len(D)):
        attractPoints = []          #存放密度吸引子吸引的点
        attractorPointsLabel = []   #存放密度吸引子吸引的点的种类标签
        densityAttr,attrRadius = findAttractor(D[i],D,h,p)  #找到密度吸引子及其半径
        #print(densityAttr,attrRadius)
        densityAttr_str = ''.join(np.array2string(densityAttr, separator=',').splitlines())  # 将densityAttr格式改为str，才能使其为字典的键
        if(densityAttr_str in densityAttractorCollection):      #可能会出现找到的密度吸引子已经存在于密度吸引子集合中的情况，若为true
            if (densityEstimation(densityAttr, D, h) >= minDensity):    #直接将密度吸引子吸引的点加到原映射中
                attractorPointsDict[densityAttr_str].append(D[i])
                attractorPointsLabelDict[densityAttr_str].append(label[i])
        else:   #若为false
            if (densityEstimation(densityAttr, D, h) >= minDensity):
                densityAttractorCollection.append(densityAttr_str)      #将密度吸引子加入密度吸引子集合中
                attractorRadiusDict[densityAttr_str] = attrRadius       #建立密度吸引子与其对应半径的映射
                attractPoints.append(D[i])                          #存放密度吸引子吸引的点
                attractorPointsLabel.append(label[i])
                attractorPointsLabelDict[densityAttr_str] = attractorPointsLabel  #建立密度吸引子与其吸引的点的种类标签的映射
                attractorPointsDict[densityAttr_str] = attractPoints    #建立密度吸引子与其吸引的所有点的映射
    # print(densityAttractorCollection)
    # print(attractorRadiusDict)
    # print(attractorPointsDict)
    # for key in attractorPointsDict:
    #     print(key)
    #     print(attractorPointsDict[key])
    attrCluster = densityReachable(densityAttractorCollection,attractorRadiusDict) #找到两两可达的吸引子的极大可达子集
    endAttractorPointsCollect = []          #分类后的点簇
    endAttractorPointsCollectLabel = []     #分类后的点簇的标签
    for k in range(len(attrCluster)):
        temp = []
        temp_label = []
        for m in range(len(attrCluster[k])):
            for l in range(len(attractorPointsDict[attrCluster[k][m]])):
                temp.append(attractorPointsDict[attrCluster[k][m]][l])
                temp_label.append(attractorPointsLabelDict[attrCluster[k][m]][l])
        endAttractorPointsCollect.append(temp)
        endAttractorPointsCollectLabel.append(temp_label)
    #print(endAttractorPointsCollect)
    #print(endAttractorPointsCollectLabel)
    #print(attractorPointsLabelDict)

    result1 = '将所有的点分为了' + str(len(attrCluster)) + '个聚类'
    print(result1)
    for j in range(len(endAttractorPointsCollect)):
        result2 = '第' + str(j+1) + '个聚类中的点数：' + str(len(endAttractorPointsCollect[j]))
        result3 = '第' + str(j+1) + '个聚类中的密度吸引子有：' + str(attrCluster[j])
        result4 = '第'+ str(j+1) +'个聚类中的点有：'+ str(endAttractorPointsCollect[j])
        print(result2)
        print(result3)
        print(result4)

    #纯度
    for n in range(len(endAttractorPointsCollectLabel)):
        a = 0
        b = 0
        c = 0
        for q in range(len(endAttractorPointsCollectLabel[n])):
            if (endAttractorPointsCollectLabel[n][q] == '"Iris-setosa"'):
                a += 1
            elif (endAttractorPointsCollectLabel[n][q] == '"Iris-versicolor"'):
                b += 1
            else:
                c += 1
        print('第' + str(n + 1) + '个聚类中各种类占比为：')
        result5 = '"Iris-setosa": ' + str(a / (len(endAttractorPointsCollect[n]))) + \
                  '"Iris-versicolor": ' + str(b / (len(endAttractorPointsCollect[n]))) + \
                  '"Iris-virginica": ' + str(c / (len(endAttractorPointsCollect[n])))
        print(result5)



if __name__ == '__main__':
    data = np.loadtxt("iris2.txt", delimiter=',', usecols=(0, 1, 2, 3))
    label = np.loadtxt("iris2.txt", str, delimiter=',', usecols=(4))
    Dat = np.array(data)
    #print(label)
    denclue(Dat,label,0.4,0.12,0.0001)
