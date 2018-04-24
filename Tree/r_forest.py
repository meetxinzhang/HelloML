import numpy as np
import random as rd

from Tree.create import create_recursion_tree
from Tree.predict import predict_test_data


def randomize_sample(dataSet, ratio):
    """
    训练数据的随机化
    创建数据集的随机子样本
    random_forest(评估算法性能，返回模型得分)
    :param dataSet 训练数据集
    :param ratio 训练数据集的样本比例
    :return sample 列表，存放随机抽样的训练样本
    """
    sample = list()
    # 训练样本的按比例抽样。
    # round() 方法返回浮点数x的四舍五入值。
    n_sample = round(len(dataSet) * ratio)
    while len(sample) < n_sample:
        # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此为自助采样法。从而保证每棵决策树训练集的差异性
        index = rd.randrange(len(dataSet))
        sample.append(dataSet[index])
    return sample


# 对随机森林的特征选择理解错误，弃用
# def randomize_features(dataSet, n_features):
#     """
#     训练特征的随机化
#     找出分割数据集的最优特征
#     :param
#         dataSet:          数据集, list
#     :param
#         n_features:       要随机选取的特征个数
#     :return:
#         alues.tolist()    最优的特征 index，特征值 row[index]， 分割完的数据 groups（left, right）
#     """
#     m, n = np.shape(dataSet)
#
#     dataSet = np.mat(dataSet)
#     values = dataSet[:, 0]
#     features_index = list()
#
#     while len(features_index) < n_features:
#         index = round(np.random.uniform(1, n-1))
#         if index not in features_index:
#             features_index.append(index)
#             values = np.c_[values, dataSet[:, index]]
#     return values.tolist()


def cross_validation_split(dataSet, n_folds):
    """
    样本数据随机无放回抽样，用于交叉验证
    将数据集进行抽重抽样 n_folds 份，数据可以重复抽取

    :param dataSet 原始数据集
    :param n_folds 数据集分成 n_flods 份
    :return dataset_split 列表
    """
    dataSet_split = list()
    dataSet_copy = list(dataSet)       # 复制一份 dataset,防止 dataset 的内容改变
    fold_size = len(dataSet) / n_folds
    for i in range(n_folds):
        fold = list()                  # 每次循环 fold 清零，防止重复导入 dataset_split
        while len(fold) < fold_size:   # 这里不能用 if，if 只是在第一次判断时起作用，while 执行循环，直到条件不成立
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此为自助采样法。从而保证每棵决策树训练集的差异性
            index = rd.randrange(len(dataSet_copy))
            # 将对应索引 index 的内容从 dataset_copy 中导出，并将该内容从 dataset_copy 中删除。
            # pop() 出栈操作, 默认是最后一个元素
            fold.append(dataSet_copy.pop(index))  # 无放回的方式
            # fold.append(dataSet_copy[index])  # 有放回的方式
        dataSet_split.append(fold)
    # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证
    return dataSet_split


def random_forest(train_data, ratio, n_tree):
    """
    创建随机森林
    :param train_data 训练数据集
    :param ratio 训练数据集的样本比例
    :param n_tree 树的个数
    :return forest 树结构，列表
    """

    forest = list()
    # n_trees 表示决策树的数量
    for i in range(n_tree):
        # [1] 样本随机， 随机采样保证了每棵决策树训练集的差异性
        sample = randomize_sample(train_data, ratio)
        # 创建一个决策树
        tree = create_recursion_tree(sample,
                                     tree_type='regression',
                                     # [2] 特征随机，在选择分割点的时候随机去掉几个特征
                                     num_remove=5,
                                     opt={'err_tolerance': 1, 'n_tolerance': 901})
        forest.append(tree)
    return forest


def random_forest_predict(forest, test_data):
    m, n = np.shape(test_data)

    yHats = np.mat(np.zeros([m, 1], float))

    for tree in forest:
        yHat = predict_test_data(tree,
                                 test_data,
                                 tree_type='regression')

        yHats = np.c_[yHats, yHat]
    # 综合所有树的预测值
    # 采取等权值投票，即均值
    pre_value = np.sum(yHats, axis=1)/10
    print(np.shape(pre_value))
    print(pre_value)







