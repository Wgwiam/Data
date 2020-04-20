import numpy as np
from numpy import *

#计算核矩阵
def getKernelMat(data):
    a = []
    for i in range(len(data)):
        b = []
        for j in range(len(data)):
            K = pow((np.dot(data[i,:],data[j,:])),2)
            b.append(K)
        a.append(b)
    KernelMat = np.mat(a)
    print("核矩阵：")
    print(KernelMat)

    #核矩阵的中心化
    unitMat = np.mat(np.identity(150))
    #print(unitMat)
    nnMat = 1/150 *mat(ones((150, 150)))
    #print(nnMat)
    centeredMat = (unitMat - nnMat) * KernelMat * (unitMat - nnMat)
    print("中心化后的核矩阵：")
    print(centeredMat)
    #核矩阵的归一化
    normalizedMat = np.mat(zeros((150,150)))
    for p in range(len(centeredMat)):
        for q in range(len(centeredMat)):
            normalizedMat[p,q] = centeredMat[p,q]/sqrt(centeredMat[p,p]*centeredMat[q,q])
    print("归一化后的核矩阵：")
    print(normalizedMat)

def transform(data):
    b = []
    for i in range(len(data)):
        a = [data[i][0]**2,data[i][1]**2,data[i][2]**2,data[i][3]**2,sqrt(2)*data[i][0]*data[i][1],sqrt(2)*data[i][0]*data[i][2],
             sqrt(2)*data[i][0]*data[i][3],sqrt(2)*data[i][1]*data[i][2],sqrt(2)*data[i][1]*data[i][3],sqrt(2)*data[i][2]*data[i][3]]
        b.append(np.array(a))
    featureSpace = np.array(b)  #变换到高维空间
    #print("变换到高维空间：")
    #print(featureSpace)
    MeanVector = np.mean(featureSpace, axis=0)
    centeredPoints = featureSpace - MeanVector      #中心化
    #print("中心化：")
    #print(centeredPoints)
    normalizedPoints = mat(ones((150,10)))
    for j in range(len(centeredPoints)):
        length = sqrt(np.dot(centeredPoints[j],centeredPoints[j]))
        normalizedPoints[j] = mat(centeredPoints[j])/length         #归一化
    #print("归一化：")
    #print(normalizedPoints)
    return normalizedPoints

def verifyKernelMat(data):
    b = []
    for i in range(len(data)):
        a = []
        for j in range(len(data)):
            InnerProduct = np.dot(data[i], data[j])  # 计算属性列间的内积
            a.append(InnerProduct)  # 一维数组
        b.append(a)  # 二维数组
    covMatrix = np.array(b)
    print(covMatrix)


if __name__ == '__main__':
    iris_data = np.loadtxt("iris.txt", skiprows=1, delimiter=' ', usecols=(1,2,3,4))
    #iris_data1 = np.loadtxt("iris2.txt", delimiter=',', usecols=(0,1, 2, 3))
    data = np.array(iris_data)
    #data1 = np.array(iris_data1)
    #getKernelMat(data)
    normalizedPoints = transform(data)
    #print(normalizedPoints)
    data2 = np.array(normalizedPoints)
    verifyKernelMat(data2)