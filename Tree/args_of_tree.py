import numpy as np


def split_data(X_train, y_train, feat_idx, value):
    """
    根据给定的特征编号和特征值对数据集进行分割
    :param X_train np.mat
    :param y_train np.mat
    :param feat_idx 待分割特征位置
    :param value 待分割的特征值
    """
    X_left, X_right, y_left, y_right = [], [], [], []

    # for line in X_train:
    #     if line[feat_idx] <= value:
    #         left.append(line)
    #     else:
    #         right.append(line)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    for row in range(0, len(y_train)):
        if X_train[row, feat_idx] <= value:
            X_left.append(X_train[row, :])
            y_left.append(y_train[row])
        else:
            X_right.append(X_train[row, :])
            y_right.append(y_train[row])

    return X_left, X_right, y_left, y_right


def choose_best_feature(X_train, y_train, tree_type='regression', num_remove=0, opt=None):
    """
    选取最佳分割特征和特征值
    :param opt: [err_tolerance: 最小误差下降值, n_tolerance: 数据切分最小样本数]
    :param num_remove 待随机去掉的特征数目
    :param tree_type 树类型
    :param dataList 待划分的数据集
    :return best_feat_idx: 最佳样本分割列
    :return best_feat_val： 最佳样本分割值
    """
    # 赋初始值
    X_train = np.array(X_train)
    m, n = X_train.shape

    if opt is None:
        opt = {'err_tolerance': 1, 'n_tolerance': 4}
    else:
        err_tolerance, n_tolerance = opt['err_tolerance'], opt['n_tolerance']

    if tree_type == 'regression':
        leaf_faction = leaf_lmTree
        err_faction = err_lmTree
    else:
        # TODO 换成其他树的叶子生成算法，和切割点衡量算法
        leaf_faction = leaf_lmTree
        err_faction = err_lmTree

    best_feat_idx, best_feat_val, best_err = 0, 0, float('inf')
    err = err_faction(X_train, y_train)

    # 随机森林部分，随机去掉 num_remove 个特征
    remove_idx = []
    if num_remove != 0:
        while len(remove_idx) < num_remove:
            # 生成 [0, n) 之间的随机整数
            index = np.random.randint(0, n)
            if index not in remove_idx:
                remove_idx.append(index)

    # 遍历所有特征，如果是随机森林，则随机去掉特征
    for feat_idx in range(0, n):  # 生成 [0,n) 的列表
        if feat_idx not in remove_idx:

            values = X_train[:, feat_idx]
            # 遍历所有特征值
            for val in values:
                # 按照当前特征和特征值分割数据
                X_left, X_right, y_left, y_right = split_data(X_train, y_train, feat_idx, val)

                if len(y_left) < n_tolerance or len(y_right) < n_tolerance:
                    # 如果切分的样本量太小，退出当前循环
                    continue

                # 计算误差
                new_err = err_faction(X_left, y_left) + err_faction(X_right, y_right)
                if new_err < best_err:
                    best_feat_idx = feat_idx
                    best_feat_val = val
                    best_err = new_err
        else:
            continue

    # 如果误差变化并不大归为一类
    if abs(err - best_err) < err_tolerance:
        return None, leaf_faction(X_train, y_train)

    # 检查分割样本量是不是太小
    X_l, X_r, y_l, y_r = split_data(X_train, y_train, best_feat_idx, best_feat_val)
    if len(y_l) < n_tolerance or len(y_r) < n_tolerance:
        return None, leaf_faction(X_train, y_train)

    return best_feat_idx, best_feat_val


def linear_regression(X_train, y_train):
    """
    获取线性回归系数
    因变量在第0列，其余为自变量
    :param dataList: 数据集
    :return w 回归系数，是一维矩阵
    :return X 自变量矩阵
    :return y 因变量矩阵
    """
    X_ori = np.matrix(X_train)
    # 这里转置是因为：转换为矩阵后形状变为了(1, n_samles)
    y_train = np.matrix(y_train)

    # 给 X_ori 添加常数列 到第一列
    m, n = X_ori.shape
    X = np.matrix(np.ones((m, n+1)))
    X[:, 1:] = X_ori

    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的不可逆，会造成程序异常
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of opt')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    w = xTx.I * (X.T * y_train)

    # print('线性回归：')
    # print(w)
    # print(X.shape)
    # print(X)
    # print(y.shape)
    # print(y)
    return w, X


def leaf_lmTree(X_train, y_train):
    """
    计算给定数据集的线性回归系数
    :param dataList: 数据集
    :return: 见 def linear_regression
    """
    w, _ = linear_regression(X_train, y_train)
    return w


def err_lmTree(X_train, y_train):
    """
    对给定数据集进行回归并计算误差
    :param dataList: 数据集
    :return: 见 def linear_regression
    """
    w, X = linear_regression(X_train, y_train)
    y_prime = X*w
    return np.var(y_prime - y_train)


