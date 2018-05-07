import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sklearn
import copy


def createData():
    data = [
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0]
    ]

    labels = ['红', '大', '甜', '脆', '贵', '好苹果']

    return data, labels


def calcEntropy(data):
    probs = {}
    for x in data:
        probs[x[-1]] = probs.get(x[-1], 0) + 1
    entropy = 0.
    for item in probs:
        probs[item] /= len(data)
        entropy -= probs[item] * np.log2(probs[item])
    result = {'probs': probs, 'entropy': entropy}
    return result


def splitData(data, index, value):
    reconstructData = []

    for x in data:
        if x[index] == value:
            d = x[:index]
            d.extend(x[index + 1:])
            reconstructData.append(d)
    return reconstructData


def conditionEntropy(data, index):
    # 统计该特征下各类别的数量和比例(概率)
    features = {}
    for x in data:
        features[x[index]] = features.get(x[index], 0) + 1

    cEntropy = 0.
    for item in features:
        features[item] /= len(data)
        tempData = splitData(data, index, item)
        result = calcEntropy(tempData)
        cEntropy += features[item] * result['entropy']
    return cEntropy


def gainEntropy(data, index):
    return calcEntropy(data)['entropy'] - conditionEntropy(data, index)


def chooseMaxClass(data):
    '''
    选择标签类中实例数最多的一类
    '''
    probs = {}
    for x in data:
        probs[x[-1]] = probs.get(x[-1], 0) + 1
    entropy = 0.
    for item in probs:
        probs[item] /= len(data)
    sortedProbs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return sortedProbs[0][0]


def chooseBestLabel(data, labels):
    '''
    选择信息增益最大的特征
    '''

    gains = {}
    for i in range(len(labels) - 1):
        gains[labels[i]] = gains.get(labels[i], 0) + gainEntropy(data, i)
    sortedGains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    return sortedGains[0]  # ( 特征, 信息增益)


def createDT(data, labels, bias=0.01):
    # 若所有数据类别相同则为单节点树
    if [x[-1] for x in data].count(data[0][-1]) == len(data):
        return data[0][-1]

    # 若没有任何特征说明，则以数据data中类别数量最多的为标记
    elif len(labels) == 1:
        return chooseMaxClass(data)

    else:
        maxGain = chooseBestLabel(data, labels)
        if maxGain[1] < bias:
            return chooseMaxClass(data)
        else:

            decisionTree = {maxGain[0]: {}}
            # 找到该特征所对应的索引
            for i in range(len(labels)):
                if labels[i] == maxGain[0]:
                    index = i
                    break

            del (labels[index])
            # 统计该特征下有多少可能值
            featureList = Counter([x[index] for x in data]).most_common()  # Counter作用可参考下面的示例。
            for i in range(len(featureList)):
                chilidData = splitData(data, index, featureList[i][0])
                decisionTree[maxGain[0]][featureList[i][0]] = createDT(chilidData, labels)

            return decisionTree


def predict(data, labels, tree):
    for key in tree:
        t = tree[key][data[labels.index(key)]]
        if type(t) != dict:
            return t
        else:
            return predict(data, labels, t)


data, labels = createData()
print(data)
print(labels)
temp_labels = copy.deepcopy(labels)
myTree = createDT(data, temp_labels)
print(myTree)
print(labels)

x = [1, 0, 1, 0, 1, 1]
l = [i for i in range(len(labels))]
print(predict(x, labels, myTree))
