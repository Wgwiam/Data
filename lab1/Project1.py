import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

#计算多元均值向量
def getMeanVector(data):
    dat = np.array(data)
    MeanVector = np.mean(dat, axis=0)
    #print(MeanVector)
    return MeanVector

#通过计算内积计算样本协方差矩阵
def getInnerProduct(data1):
    b = []
    for i in range(len(data1[0])):
        a = []
        for j in range(len(data1[0])):
            InnerProduct = np.dot(data1[:,i],data1[:,j]) #计算属性列间的内积
            a.append(InnerProduct)  #一维数组
        b.append(a) #二维数组
    covMatrix = np.array(b)/(len(data1)-1) #样本无偏的协方差
    return covMatrix

#通过计算外积计算样本协方差矩阵
def getOuterProduct(data1):
    #dat = np.mat(data1)
    covMatrix = np.mat(zeros((10,10)))  #创建一个10*10的零矩阵
    for i in range(len(data1)):
        OuterProduct = (np.mat(data1[i, :]).T) * np.mat(data1[i, :]) #计算中心点的张量积
        covMatrix = covMatrix + OuterProduct    #矩阵累加
    covMatrix1 = covMatrix/(len(data1)-1)  #样本无偏的协方差
    return covMatrix1

    #print((np.mat(data1[0, :]).T) * np.mat(data1[0, :]))

def getCorrelation(data1):
    cov12 = np.dot(data1[:, 0], data1[:, 1])/(len(data1)-1)  # 计算属性1和属性2的协方差
    cov1cov2 = sqrt(dot(data1[:, 0], data1[:, 0]) * np.dot(data1[:, 1], data1[:, 1])/pow((len(data1)-1),2)) #计算属性1和属性2方差的乘积
    correlation = cov12/cov1cov2
    print(correlation)

def drawScatter(data2):
    # 创建画布
    fig = plt.figure()
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    # 设置可视化图的标题等属性
    ax.set_title("The scatter plot between Attributes 1 and 2")

    x = data2[:, 0]
    y = data2[:, 1]
    # 绘制散点图
    plt.scatter(x, y, s=10, marker='o', c='grey')

    # 隐藏绘图区的顶部及右侧坐标轴
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    # 设置坐标title
    ax.axis["left"].label.set_text("X2:Attributes 2")
    ax.axis["bottom"].label.set_text("X1:Attributes 1")
    # 设置绘图区的底部及左侧坐标轴样式 "-|>"代表实心箭头："->"代表空心箭头
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    # 设置坐标轴数值显示的方向
    ax.axis["left"].set_axis_direction("left")
    # 显示绘制结果
    plt.show()

def drawNormal(data3):
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.set(ylabel='f(x)', xlabel='Normal distribution:probability density function')

    # 求均值
    data_mean = np.mean(data3)
    # 求方差
    data_var = np.var(data3)
    # 求标准差
    data_std = np.std(data3)
    print(data_mean)
    print(data_var)

    x = np.linspace(data_mean - 5 * data_std, data_mean + 5 * data_std, 19020)
    y_sig = np.exp(-(x - data_mean) ** 2 / (2 * data_std ** 2)) / (math.sqrt(2 * math.pi) * data_std)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    ax.axis["bottom"] = ax.new_floating_axis(0, np.min(y_sig))
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axis_direction("left")

    new_ticks = np.linspace(-150, 250, 9) #设置x轴间隔
    plt.xticks(new_ticks)

    plt.plot(x, y_sig, "black", linewidth=2)
    plt.show()

def CompareVar(data):
    all_var = []
    dict = {}
    for i in range(len(data1[0])):
        data_var = np.var(data[:,i])
        all_var.append(data_var) #加入list
        dict[i] = data_var #加入字典
    result = sorted(dict.items(), key=lambda x: x[1])  #按字典的value排序
    maxVar = max(all_var) #list中的最大值
    maxAttr = all_var.index(max(all_var)) #list中的最大值在list中的索引
    minVar = min(all_var)
    minAttr = all_var.index(min(all_var))
    result_max = '方差最大的属性是：' + repr(maxAttr + 1) + '，其方差为：' + repr(maxVar)
    result_min = '方差最小的属性是：' + repr(minAttr + 1) + '，其方差为：' + repr(minVar)
    print(result_max)
    print(result_min)
    #print(result)

def CompareCov(covMatrix):
    covariance = np.array(covMatrix)
    min = max = covariance[0][1]
    min_attr = max_attr = '0'+'和'+'1'
    for i in range(len(covariance)):
        for j in range(len(covariance[0])):
            if(i == j):
                pass
            else:
                if covariance[i][j] < min:
                    min = covariance[i][j]
                    min_attr = str(i+1)+'和'+str(j+1)
                elif covariance[i][j] > max:
                    max = covariance[i][j]
                    max_attr = str(i+1)+'和'+str(j+1)
    result_min = '属性'+ repr(min_attr) + '的协方差最小，值为：' + repr(min)
    result_max = '属性' + repr(max_attr) + '的协方差最大，值为：' + repr(max)
    print(result_min)
    print(result_max)

if __name__=='__main__':
    data = np.loadtxt("magic04.data",delimiter=',', usecols=(0,1,2,3,4,5,6,7,8, 9))
    data2 = np.loadtxt("magic04.data", delimiter=',', usecols=(0, 1))
    data3 = data2[:,0]
    dat = np.array(data)
    #print(data)
    MeanVector = getMeanVector(data)
    data1 = dat - MeanVector
    covMatrix1 = getInnerProduct(data1)

    covMatrix2 = getOuterProduct(data1)
    #print(covMatrix2)
    #print(covMatrix2)
    #getCorrelation(data1)
    #drawScatter(data2)
    #drawNormal(data3)
    CompareVar(data)
    #CompareCov(covMatrix1)


