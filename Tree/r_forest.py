import numpy as np
import random as rd

from Tree.create import create_recursion_tree
from Tree.args import leaf_lmTree, err_lmTree
from Tree.predict import predict_test_data, predict_lmTree


def randomize_sample(dataSet, ratio):
    """
    训练数据的随机化
    创建数据集的随机子样本
    random_forest(评估算法性能，返回模型得分)
    :param
        dataSet         训练数据集
        ratio           训练数据集的样本比例
    :return:
        sample          随机抽样的训练样本
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


def randomize_features(dataSet, n_features):
    """
    训练特征的随机化
    找出分割数据集的最优特征
    :param
        dataSet:          数据集, list
    :param
        n_features:       要随机选取的特征个数
    :return:
        alues.tolist()    最优的特征 index，特征值 row[index]， 分割完的数据 groups（left, right）
    """
    m, n = np.shape(dataSet)

    dataSet = np.mat(dataSet)
    values = dataSet[:, 0]
    features_index = list()

    while len(features_index) < n_features:
        index = round(np.random.uniform(1, n-1))
        if index not in features_index:
            features_index.append(index)
            values = np.c_[values, dataSet[:, index]]
    return values.tolist()


def cross_validation_split(dataSet, n_folds):
    """
    样本数据随机无放回抽样，用于交叉验证
    将数据集进行抽重抽样 n_folds 份，数据可以重复抽取

    :param
        dataset          原始数据集
        n_folds          数据集dataset分成n_flods份
    :return
        dataset_split    list集合，存放的是：将数据集进行抽重抽样 n_folds 份，数据可以重复抽取
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


# Random Forest Algorithm
def random_forest(train_data, ratio, n_tree, n_features):
    """
    创建随机森林
    :param
        train           训练数据集
        ratio           训练数据集的样本比例
        n_trees         决策树的个数
        n_features      选取的特征的个数
    :return
        predictions     每一行的预测结果，bagging 预测最后的分类结果
    """

    forest = list()
    # n_trees 表示决策树的数量
    for i in range(n_tree):
        # 随机抽样的训练样本， 随机采样保证了每棵决策树训练集的差异性
        sample = randomize_sample(train_data, ratio)
        #sample = randomize_features(sample, n_features)
        # 创建一个决策树
        tree = create_recursion_tree(sample,
                                     leaf_faction=leaf_lmTree,
                                     err_faction=err_lmTree,
                                     opt={'err_tolerance': 1, 'n_tolerance': 901})
        forest.append(tree)
    return forest


def random_forest_predict(forest, test_data):
    yHat_list = []

    for tree in forest:
        yHat = predict_test_data(tree,
                                 test_data,
                                 predictFacion=predict_lmTree)
        yHat_list.append(yHat)
    # 综合所有树的预测值
    # 取平均值
    # TODO
    






