from sklearn import metrics
from PCA.main import outPut
import numpy as np
import random as rd


def mre(y_true, y_pre):
    score = metrics.mean_absolute_error(y_true, y_pre, multioutput='uniform_average')
    print('平均绝对误差为：\n', score)
    pass


def r2(y_true, y_pre):
    score = metrics.r2_score(y_true, y_pre, multioutput='uniform_average')
    print('决定系数 R^2 为：\n', score)
    pass


def load_list_data_and_standardization(filename, n_folds, n_test=1, idx_test=0):
    """
    加载本地数据，返回 list 类型
    使用交叉验证：
        原数据进行随机不放回抽样，分成 n_folds 等份，其中 n_test 份用作测试，其余的用作训练。
        在外部调用本函数 (n_folds-n_test)+1 次，每一次 idx_test 增加1，增加到 n_folds-n_test 为止，
        从而让每一份都作为测试数据一次，这就是 n_folds-折交叉验证。
    :param filename: 文件名
    :param n_folds: 交叉验证分成的份数，一般为10
    :param n_test: 测试数据所占的份数，一般为1
    :param idx_test: 第一份测试数据所在的起点位置，从0-9遍历
    :return:
        X_train1, y_train1, X_test1 训练数据，训练标签，测试数据，都是list类型
    """
    if n_test >= n_folds/2:
        raise NameError('测试样本超过了总样本的一半，请减小 n_test 参数，一般设置为1')
    if idx_test + n_test > n_folds:
        raise NameError('测试样本的起点设置越界')

    m = 0
    dataList = []
    with open(filename, 'r') as f:
        for line in f:
            m = m+1
            line_data = [float(data) for data in line.split('\t')]
            dataList.append(line_data)

    # # 不相关处理
    # dataList = np.array(dataList)
    # dataList[:, 5] = dataList[:, 5] / dataList[:, 6]
    # dataList = np.delete(dataList, [2, 6], axis=1).tolist()

    # 检验缺省值
    print('缺省检验：', np.isnan(dataList).any())
    print('原数据共{}行，随机分成{}等份，第{}份作为测试样本'.format(m, n_folds, idx_test+1))

    dataSet_split = cross_validation_split(dataList, n_folds)

    train_data = []
    test_data = []

    for i in range(n_folds):
        if i < idx_test + n_test:
            test_data.append(dataSet_split[i])
            # np.concatenate([test_data, dataSet_split[i]], axis=0)
            # np.r_[test_data, dataSet_split[i]]
        else:
            train_data.append(dataSet_split[i])
            # np.concatenate([train_data, dataSet_split[i]], axis=0)
            # np.r_[train_data, dataSet_split[i]]

    train_data = np.concatenate(train_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    train_data = np.matrix(train_data)
    test_data = np.matrix(test_data)

    X_test1 = test_data[:, 1:]
    #X_test1 = test_data[:, :-1]
    X_test1 = X_test1.tolist()
    X_test1 = standardization(X_test1)

    y_test1 = test_data[:, 0].tolist()
    #y_test1 = test_data[:, -1].tolist()

    X_train1 = train_data[:, 1:]
    #X_train1 = train_data[:, :-1]
    X_train1 = X_train1.tolist()
    X_train1 = standardization(X_train1)

    y_train1 = train_data[:, 0].tolist()
    #y_train1 = train_data[:, -1].tolist()

    return X_train1, y_train1, X_test1, y_test1


def cross_validation_split(dataSet, n_folds):
    """
    样本数据随机无放回抽样，用于交叉验证
    将数据集进行抽重抽样 n_folds 份，数据可以重复抽取

    :param dataSet 原始数据集
    :param n_folds 数据集分成 n_flods 份
    :return dataset_split 列表
    """
    dataSet_split = list()
    dataSet_copy = dataSet      # 复制一份 dataset,防止 dataset 的内容改变
    fold_size = len(dataSet) / n_folds
    for i in range(n_folds):
        fold = list()                  # 每次循环 fold 清零，防止重复导入 dataset_split
        while len(fold) < fold_size and len(dataSet_copy) > 0:   # 这里不能用 if，if 只是在第一次判断时起作用，while 执行循环，直到条件不成立
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此为自助采样法。从而保证每棵决策树训练集的差异性
            index = rd.randrange(0, len(dataSet_copy), 1)
            # 将对应索引 index 的内容从 dataset_copy 中导出，并将该内容从 dataset_copy 中删除。
            # pop() 出栈操作, 默认是最后一个元素
            fold.append(dataSet_copy.pop(index))  # 无放回的方式
            # fold.append(dataSet_copy[index])  # 有放回的方式
        dataSet_split.append(fold)
    # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证
    return dataSet_split


def usePCA(dataX):
    # 使用 PCA 将数据变为线性无关，影响预测准确性，弃用
    # 分割出自变量，因为PCA只处理自变量
    _, n = np.shape(dataX)

    pca_X = outPut(dataX, n)

    return pca_X.tolist()


def standardization(dataX):
    """
    0均值标准化(Z-score standardization)
    因为数据本身的量纲不统一，导致数据的数量级差别很大，导致测试数据的过拟合问题很严重，
    所以需要使用标准化处理。
    :param dataX 输入数据，列表
    :return dataTad.tolist()
    """
    dataX = np.array(dataX)
    # 我们的数据变量按列进行排列(即一行为一个样本),按列求均值，即求各个特征的均值
    meanVal = dataX.mean(axis=0)
    # meanVal = np.mean(dataX, axis=0) 此同为np的方法,得到Series
    # 求标准差
    stdVal = dataX.std(axis=0)
    dataTad = (abs(dataX-meanVal))/stdVal
    return dataTad.tolist()




