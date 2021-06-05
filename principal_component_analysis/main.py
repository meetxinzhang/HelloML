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
    mean_val = dataX.mean(axis=0)
    # mean_val = np.mean(dataX, axis=0) 此同为np的方法,得到Series
    # 求标准差
    std_val = dataX.std(axis=0)
    data_tad = (dataX - mean_val) / std_val
    return data_tad


def pcan(dataX, data_tad, n):
    # 协方差矩阵
    data_cov = data_tad.cov()
    # 相关系数矩阵
    # data_cov = datasTad.corr()

    new_data1 = np.array(data_cov)
    # 求得特征值，特征向量
    eigen_value, eigen_vector = np.linalg.eig(new_data1)
    # 特征值下标从小到大的排列顺序
    source_eigen_value = np.argsort(eigen_value)
    # 最大的n个特征值的下标
    n_pca_eigen_vector = source_eigen_value[-n:]
    # 选取特征值对应的特征向量
    pca_eigen_vector = eigen_vector[n_pca_eigen_vector]
    # 得到降维后的数据
    pcax = np.dot(dataX, pca_eigen_vector.T)
    return pcax, pca_eigen_vector


def load_data(filename):
    """
    加载本地数据文件，
    :param filename: 文件地址
    :return: list 矩阵
    """
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            line_data = [float(data) for data in line.split('\t')]
            data_list.append(line_data)

    print(np.shape(data_list))
    return data_list


def output(data_list, dim):
    # 导入数据，切记不含因变量。我们在此构造df1数据，此数据变量间没有一定的相关性，只做计算演示。
    df1 = np.array(data_list)

    # # 去掉第一列因变量
    # df1 = df1[:, 1:]

    df1 = pd.DataFrame(df1)
    data_tad = standardization(df1)
    # 选取主成份
    pcax, pca_eigen_vector = pcan(df1, data_tad, dim)

    print('选取的特征向量：')
    print(pca_eigen_vector, end='\n')
    print('降维后的数据：')
    print(pcax, end='\n')
    return pcax


if __name__ == "__main__":
    # 导入数据，切记不含因变量。我们在此构造df1数据，此数据变量间没有一定的相关性，只做计算演示。
    df1 = np.array(load_data('data.txt'))

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
