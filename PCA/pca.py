import numpy as np


def pca(X, k):
    """
    :param X: 数据集 list
    :param k: 想要的维度
    :return: 新的特征矩阵
    """
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)

    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    print('特征值为：')
    print(eig_val)
    print('特征向量：')
    print(eig_vec)

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    newData = np.dot(norm_X, np.transpose(feature))
    return newData


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


data = np.array(load_data('train_data.txt'))
# 去掉第一列因变量
# data = data[:, 1:]
x_ = pca(data, 1)

print('降维数据：')
print(x_)
