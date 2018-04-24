import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def standardization(dataX):
    """
    0均值标准化(Z-score standardization)
    :param dataX:
    :return:
    """
    # 我们的数据变量按列进行排列(即一行为一个样本),按列求均值，即求各个特征的均值
    meanVal = dataX.mean(axis=0)
    # meanVal = np.mean(dataX, axis=0) 此同为np的方法,得到Series
    # 求标准差
    stdVal = dataX.std(axis=0)
    datasTad = (dataX-meanVal)/stdVal
    return datasTad


def pcan(dataX, datasTad, n):
    # 协方差矩阵
    dataCov = datasTad.cov()
    # 相关系数矩阵
    # dataCov = datasTad.corr()

    newData1 = np.array(dataCov)
    # 求得特征值，特征向量
    eigenValue, eigenVector = np.linalg.eig(newData1)
    # 特征值下标从小到大的排列顺序
    sorceEigenValue = np.argsort(eigenValue)
    # 最大的n个特征值的下标
    nPcaEigenVector = sorceEigenValue[-n:]
    # 选取特征值对应的特征向量
    pcaEigenVector = eigenVector[nPcaEigenVector]
    # 得到降维后的数据
    PCAX = np.dot(dataX, pcaEigenVector.T)
    return PCAX, pcaEigenVector


def load_data(filename):
    """
    加载本地数据文件，
    :param filename: 文件地址
    :return: list 矩阵
    """
    dataList = []
    with open(filename, 'r') as f:
        for line in f:
            line_data = [float(data) for data in line.split('\t')]
            dataList.append(line_data)

    print(np.shape(dataList))
    return dataList


def outPut(dataList, dim):
    # 导入数据，切记不含因变量。我们在此构造df1数据，此数据变量间没有一定的相关性，只做计算演示。
    df1 = np.array(dataList)

    # # 去掉第一列因变量
    # df1 = df1[:, 1:]

    df1 = pd.DataFrame(df1)
    datasTad = standardization(df1)
    # 选取主成份
    PCAX, pcaEigenVector = pcan(df1, datasTad, dim)

    print('选取的特征向量：')
    print(pcaEigenVector, end='\n')
    print('降维后的数据：')
    print(PCAX, end='\n')
    return PCAX


if __name__ == "__main__":
    # 导入数据，切记不含因变量。我们在此构造df1数据，此数据变量间没有一定的相关性，只做计算演示。
    df1 = np.array(load_data('train_data.txt'))
    # 去掉第一列因变量
    # df1 = df1[:, 1:]

    df1 = pd.DataFrame(df1)
    datasTad = standardization(df1)
    # 选取主成份
    PCAX, pcaEigenVector = pcan(df1, datasTad, 5)

    print('选取的特征向量：')
    print(pcaEigenVector, end='\n')
    print('降维后的数据：')
    print(PCAX, end='\n')

    # 与 sklearn 框架的PCA 对比
    pca = PCA(n_components=5)
    new_pca = pd.DataFrame(pca.fit_transform(datasTad))
    print('\n')
    print('sklearn :')
    print(new_pca)
pass
