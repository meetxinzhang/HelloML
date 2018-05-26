from Tree.args_of_tree import *


class MyTree:

    # 预剪枝时用到的超参数，最小允许误差，最小允许样本数
    opt = {'err_tolerance': 1, 'n_tolerance': 4}
    # 生长节点时随机去掉的特征数量，在构建随机森科时可能需要用到
    num_remove_feature = 0
    # 树的类型
    tree_type = 'regression'
    # 树模型字典
    # feat_idx: 内部节点对应在数据上的列位置，表示叶节点时为None
    # feat_val: 内部节点值，表示叶节点时为回归系数矩阵
    # left: 左子树指针
    # right: 右子树指针
    tree = {'feat_idx': None, 'feat_val': None, 'left': None, 'right': None}

    def __init__(self, tree_type='regression', num_remove_feature=0, opt=None):
        self.num_remove_feature = num_remove_feature
        self.tree_type = tree_type
        if opt is None:
            self.opt = {'err_tolerance': 1, 'n_tolerance': 4}
        else:
            self.opt = opt

    def fit(self, X_train, y_train):
        self.tree = self.recursion_create_tree(X_train, y_train)
        return self.tree

    def recursion_create_tree(self, X_train, y_train):
        """
        创建迭代树结构
        :return tree 树模型参数，字典类型
        """
        # 选择最优化分特征和特征值
        feat_idx, value = choose_best_feature(X_train, y_train,
                                              self.tree_type,
                                              self.num_remove_feature,
                                              self.opt)

        while feat_idx == 'err':
            self.opt['n_tolerance'] += 10
            print('matrix is singular, cannot do inverse,\nincreasing the second value of opt to {}'
                  .format(self.opt['n_tolerance']))
            feat_idx, value = choose_best_feature(X_train, y_train,
                                                  self.tree_type,
                                                  self.num_remove_feature,
                                                  self.opt)
            if feat_idx != 'err':
                break

            print('1111111111111: ', feat_idx)

        # 触底条件: 该点划分为叶子节点，此时 value 为回归系数矩阵
        if feat_idx is None:
            return value

        # 创建一层树结构
        tree = {'feat_idx': feat_idx, 'feat_val': value}

        # 递归创建左子树和右子树
        X_left, X_right, y_left, y_right = split_data(X_train, y_train, feat_idx, value)
        ltree = self.recursion_create_tree(X_left, y_left)
        rtree = self.recursion_create_tree(X_right, y_right)
        # 指向左右子树
        tree['left'] = ltree
        tree['right'] = rtree

        return tree

    def predict_1X_on_linear_moedl_leaf(self, model, one_x):
        """
        预测，一行测试数据 * 回归系数
        对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
        也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
        :param model -- 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型，实则为 回归系数
        :param one_x -- 输入的测试数据
        :return float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
        """
        one_x = np.matrix(one_x)
        n = np.shape(one_x)[1]
        X = np.mat(np.ones((1, n + 1)))
        X[:, 1: n + 1] = one_x

        return float(X * model)

    def predict_1X_on_tree(self, tree, one_x):
        """
        迭代遍历树
        :param one_x -- 输入的测试数据，只有一行
        :return 返回预测值
        """
        if self.tree_type == 'regression':
            predict_faction = self.predict_1X_on_linear_moedl_leaf
        else:
            # TODO 换成其他树类型的预测函数
            predict_faction = self.predict_1X_on_linear_moedl_leaf

        if self.isLeaf(tree):
            return predict_faction(tree, one_x)

        # 书中写的是inData[tree['spInd']]，只适合inData只有一列的情况，否则会产生异常

        # one_x = np.matrix(one_x).T.tolist()
        if one_x[tree['feat_idx']] <= tree['feat_val']:
            # 可以把if-else去掉，只留if里面的分支
            # if not self.isLeaf(tree['left']):
                return self.predict_1X_on_tree(tree['left'], one_x)
            # else:
            #     return predict_faction(tree['left'], one_x)
        else:
            # 同上，可以把if-else去掉，只留if里面的分支
            # if not self.isLeaf(tree['right']):
                return self.predict_1X_on_tree(tree['right'], one_x)
            # else:
            #     return predict_faction(tree['right'], one_x)

    def predict(self, X_test):
        """
        计算全部测试结果
        调用 predict_lmTree ，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        :param
            tree -- 已经训练好的树的模型
            testData -- 输入的测试数据
            modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
        :return
            返回预测值矩阵
        """
        m = len(X_test)
        yHat = np.mat(np.zeros((m, 1)))
        # print yHat
        for i in range(m):
            yHat[i, 0] = self.predict_1X_on_tree(self.tree, X_test[i])
            # print('yHat==>', yHat[i, 0])
        return yHat

    # # 判断节点是否是一个字典
    # def isTree(self, obj):
    #     """
    #     测试输入变量是否是一棵树,即是否是一个字典
    #     :param
    #         obj -- 输入变量
    #     :return
    #         返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
    #     """
    #     return type(obj).__name__ == 'dict'

    def isLeaf(self, obj):
        """
        判断节点是否为叶子节点
        :param obj: 输入节点
        :return: Boolean
        """
        # 字典类型是内部节点，矩阵类型（回归系数）是叶子节点
        return not isinstance(obj, dict)
