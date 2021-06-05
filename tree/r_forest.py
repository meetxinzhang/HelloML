import numpy as np
from numba import jit
import random as rd

from tree.my_tree import MyTree


class MyRandomForest:
    tree_type = 'regression',
    num_remove_feature = 0,
    opt = {'err_tolerance': 1, 'n_tolerance': 4},
    sample_ratio = 0.7,
    n_tree = 100

    # 存放 my_tree.py 里 MyTree 类的对象
    forest = list()
    # 森林结构，为所有树结构的平均值
    # struct = list()

    def __init__(self,
                 tree_type='regression',
                 num_remove_feature=5,
                 opt=None,
                 sample_ratio=0.7,
                 n_tree=100):
        """
        :param tree_type: 树类型，目前只支持LMT
        :param num_remove_feature: 构建树的时候随机去掉的特征数目
        :param opt: 预剪枝用到的超参数，'err_tolerance': 左右子树最小允许误差，'n_tolerance'：左右子树最小允许样本数
        :param sample_ratio: 构建树的时候随机抽样所占总样本的比例
        :param n_tree: 树的数量
        """
        if opt is None:
            self.opt = {'err_tolerance': 1, 'n_tolerance': 4}
        else:
            self.opt = opt
        self.tree_type = tree_type
        self.num_remove_feature = num_remove_feature
        self.opt = opt
        self.sample_ratio = sample_ratio
        self.n_tree = n_tree

    def randomize_sample(self, X_all, y_all, sample_ratio):
        """
        随机抽取样本，使得每一棵树的训练样本都不一样
        :param X_all: 训练数据
        :param y_all: 训练标签
        :param sample_ratio: 随机抽取的样本占总样本的比例，需要调参
        :return:
        """
        X_train = list()
        y_train = list()
        # 训练样本的按比例抽样。
        # round() 方法返回浮点数x的四舍五入值。
        n_sample = round(len(y_all) * sample_ratio)
        while len(y_train) < n_sample:
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此为自助采样法。从而保证每棵决策树训练集的差异性
            index = rd.randrange(len(y_all))
            X_train.append(X_all[index])
            y_train.append(y_all[index])
        return X_train, y_train

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

    @jit()
    def fit(self, X_train, y_train):
        """
        :param X_train: 训练数据
        :param y_train: 训练标签
        :return:
            返回森林中所有树内部节点值的均值
        """
        # n_trees 表示决策树的数量
        for i in range(self.n_tree):
            # [1] 样本随机， 随机采样保证了每棵决策树训练集的差异性
            X_train_item, y_train_item = self.randomize_sample(X_train, y_train, self.sample_ratio)

            # 创建一个决策树
            tree = MyTree(tree_type=self.tree_type,
                          # [2] 特征随机，在选择分割点的时候随机去掉几个特征
                          num_remove_feature=self.num_remove_feature,
                          opt=self.opt)
            # 训练
            struct = tree.fit(X_train_item, y_train_item)
            self.forest.append(tree)

            print('该树的特征分裂计数  \n', tree.the_list)
            print('num of tree: ', len(self.forest))
            print(struct)

        # self.struct = np.concatenate(self.struct, axis=1)
        # self.struct = np.sum(self.struct, axis=1)/self.n_tree
        # return self.struct

    @jit()
    def predict(self, X_test):
        m, n = np.shape(X_test)

        yHats = np.mat(np.zeros([m, 1], float))

        for tree in self.forest:
            yHat = tree.predict(X_test)

            yHats = np.c_[yHats, yHat]
        # 综合所有树的预测值
        # 采取等权值投票，即均值
        pre_value = np.sum(yHats, axis=1) / self.n_tree
        return pre_value
    pass
