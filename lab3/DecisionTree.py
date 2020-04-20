import numpy as np

def EvaluateNumericAttribute(D,X):
    sortedIndex = np.argsort(X)   #按X的值排序
    splitPoints = []                #存放所有切分点
    numOfPoints = [0,0,0]           #初始化3类中点的个数都为0
    splitNumOfPoiDict = {}          #建立切分点与包含各类的店的映射
    for i in range(len(sortedIndex) - 1):       #遍历前n-1个点，每个点对其所属的类数量+1
        if(D[sortedIndex[i]][4] == '"Iris-versicolor"'):
            numOfPoints[0] += 1
        elif(D[sortedIndex[i]][4] == '"Iris-setosa"'):
            numOfPoints[1] += 1
        else:
            numOfPoints[2] += 1
        if(X[sortedIndex[i+1]] != X[sortedIndex[i]]): #两个连续不同值点的中点作为切分点
            splitPoi = (X[sortedIndex[i+1]] + X[sortedIndex[i]])/2
            splitPoints.append(splitPoi)
            temp = []
            for i in range(len(numOfPoints)):
                temp.append(numOfPoints[i])
            splitNumOfPoiDict[splitPoi] = temp  #保存当前切分点与它左子树包含的各类别的数量
    #print(splitNumOfPoiDict)

    #最后一个点不会作为分裂点
    if (D[sortedIndex[-1]][4] == '"Iris-versicolor"'):
        numOfPoints[0] += 1
    elif (D[sortedIndex[-1]][4] == '"Iris-setosa"'):
        numOfPoints[1] += 1
    else:
        numOfPoints[2] += 1
    bestSplitPoint = []
    splitScore = 0
    for point in splitPoints:
        for j in range(3):
            P1 = splitNumOfPoiDict[point][j]/(splitNumOfPoiDict[point][0] + splitNumOfPoiDict[point][1] + splitNumOfPoiDict[point][2])
            P2 = (numOfPoints[j] - splitNumOfPoiDict[point][j])/(numOfPoints[0] + numOfPoints[1] + numOfPoints[2]
                                                                 - splitNumOfPoiDict[point][0] - splitNumOfPoiDict[point][1] - splitNumOfPoiDict[point][2])
        # 数据集的熵
        HD = 0
        for l in range(len(numOfPoints)):
            H = numOfPoints[l] / len(D)
            if (H != 0):
                HD += (H * np.log2(H))
        HD = 0 - HD
        # 计算信息增益
        HDy = 0  # 左半空间的熵
        HDn = 0  # 右半空间的熵
        for i in range(len(numOfPoints)):
            Yprob = splitNumOfPoiDict[point][i] / np.sum(splitNumOfPoiDict[point])
            Nprob = (numOfPoints[i] - splitNumOfPoiDict[point][i]) / (np.sum(numOfPoints) - np.sum(splitNumOfPoiDict[point]))
            if Yprob != 0:
                HDy += (Yprob * np.log2(Yprob))
            if Nprob != 0:
                HDn += (Nprob * np.log2(Nprob))
        HDy = 0 - HDy
        HDn = 0 - HDn
        # 计算信息增益
        temp_score = HD - (((np.sum(splitNumOfPoiDict[point]) / len(D)) * HDy) + (
        ((len(D) - np.sum(splitNumOfPoiDict[point])) / len(D)) * HDn))
        if (temp_score > splitScore):
            bestSplitPoint = point
            splitScore = temp_score
        #print(temp_score)
        #print(point)
    #print(bestSplitPoint)
    #print(splitScore)
    return bestSplitPoint,splitScore

#创建叶子节点
class LeafNode(object):
    def __init__(self,labelType,purity,size):
        self.labelType = labelType  #叶子节点分类的标签
        self.purity = purity        #叶子节点的纯度
        self.size = size            #叶子节点的大小
        self.leftChild=None        #左节点
        self.rightChild=None       #右节点
    def __str__(self):
        return "标签：" + str(self.labelType)+ " 纯度："+str(self.purity)+" 点数："+str(self.size)

#创建内部节点
class InternalNode(object):
    def __init__(self,splitPoint,attrIndex):
        self.splitPoint = splitPoint        #分割点
        self.attrIndex = attrIndex          #属性序号
        self.leftChild=None                 #表示左节点
        self.rightChild=None                #表示右节点
    def __str__(self):
        return "分割点："+str(self.splitPoint)+" 属性列："+str(self.attrIndex)

#构建二叉树
class BinaryTree(object):
    def __init__(self):
        self.root=InternalNode('root',"∞")  #根节点定义为 root 永不删除，作为哨兵使用。

    # 添加左节点，需要给出添加的节点、父节点
    def addLeft(self, node, rootNode):
        if self.root is None:       #如果二叉树为空，则添加的节点为新构建的二叉树的根节点
            self.root = node
        else:
            rootNode.leftChild = node

    # 添加右节点，需要给出添加的节点、父节点
    def addRight(self, node, rootNode):
        if self.root is None:       #如果二叉树为空，则添加的节点为新构建的二叉树的根节点
            self.root = node
        else:
            rootNode.rightChild = node

    # 前序遍历
    def preTraverse(self,root):
        if root is None:
            return
        print(root)
        self.preTraverse(root.leftChild)
        self.preTraverse(root.rightChild)

def DecisionTree(D,numOfLeaf,minPurity,tree,root):
    numOfVersicolor = 0
    numOfSetosa = 0
    numOfVirginica = 0
    for i in range(len(D)):
        if (D[i][4] == '"Iris-versicolor"'):
            numOfVersicolor += 1
        elif (D[i][4] == '"Iris-setosa"'):
            numOfSetosa += 1
        else:
            numOfVirginica += 1
    purityOfD = max(numOfVersicolor,numOfSetosa,numOfVirginica)/len(D)
    proportionOfMajorityDict = {}
    proportionOfMajorityDict["Iris-versicolor"] = numOfVersicolor/len(D)
    proportionOfMajorityDict["Iris-setosa"] = numOfSetosa / len(D)
    proportionOfMajorityDict["Iris-virginica"] = numOfVirginica / len(D)
    #print(proportionOfMajorityDict)
    if(len(D) <= numOfLeaf or purityOfD >= minPurity):
        #用占比最大的种类作为数据集的标签
        D_label = list(proportionOfMajorityDict.keys())[list(proportionOfMajorityDict.values()).index(max(proportionOfMajorityDict.values()))]
        #创建叶子节点并将其标签标记为D_label
        leafNode = LeafNode(D_label,proportionOfMajorityDict[D_label],len(D))
        tree.addLeft(leafNode,root)
        return

    bestSplitPoint = []  #最佳分割点
    splitScore = 0       #最佳分割点对应的信息增益
    bestFeatureIndex = 0    #最佳分割点对应的属性
    #遍历所有属性，找到最佳分割点
    for i in range(len(D[0]) -1):
        temp = []
        for j in range(len(D)):
            temp.append(D[j][i])
        splitPoi,score = EvaluateNumericAttribute(D,temp)
        if(score > splitScore):
            splitScore = score
            bestSplitPoint = splitPoi
            bestFeatureIndex = i
    #将数据D划分为左右子树
    Dy = []
    Dn = []
    for j in range(len(D)):
        if(D[j][bestFeatureIndex] <= bestSplitPoint):
            Dy.append(D[j])
        else:
            Dn.append(D[j])
    leftInternalNode = InternalNode(bestSplitPoint,bestFeatureIndex)
    tree.addLeft(leftInternalNode,root)
    DecisionTree(Dy,numOfLeaf,minPurity,tree,leftInternalNode)
    RightInternalNode = InternalNode(bestSplitPoint,bestFeatureIndex)
    tree.addRight(RightInternalNode,root)
    DecisionTree(Dn, numOfLeaf, minPurity, tree, RightInternalNode)

if __name__ == '__main__':
    data = np.loadtxt("iris2.txt", delimiter=',', usecols=(0, 1, 2, 3))
    X = data[:,0]
    label = np.loadtxt("iris2.txt", str, delimiter=',', usecols=(4))
    Dat = []
    for i in range(len(data)):
        temp = []
        temp.append(data[i][0])
        temp.append(data[i][1])
        temp.append(data[i][2])
        temp.append(data[i][3])
        temp.append(label[i])
        Dat.append(temp)
    #print(np.array(Dat))
    #EvaluateNumericAttribute(Dat,X)
    #print(type(Dat))
    decTree = BinaryTree()
    DecisionTree(Dat,5,0.95,decTree,decTree.root)
    decTree.preTraverse(decTree.root)
