import numpy as np


def split_data(dataset, feat_idx, value):
    """
    根据给定的特征编号和特征值对数据集进行分割
    :returns list 矩阵
    """
    left, right = [], []
    for line in dataset:
        if line[feat_idx] <= value:
            left.append(line)
        else:
            right.append(line)

    return left, right


def choose_best_feature(dataList, leaf_faction, err_faction, opt):
    """
    选取最佳分割特征和特征值
    :param dataList: 待划分的数据集
           leafType: 创建叶子节点的函数
           errType: 计算数据误差的函数
           opt: [err_tolerance: 最小误差下降值, n_tolerance: 数据切分最小样本数]
    :returns best_feat_idx: 最佳样本分割列
             best_feat_val： 最佳样本分割值
    """
    dataList = np.array(dataList)
    m, n = dataList.shape
    err_tolerance, n_tolerance = opt['err_tolerance'], opt['n_tolerance']

    # # 如果结果集(最后一列为1个变量)，就返回退出
    # # .T 对数据集进行转置
    # # .tolist()[0] 转化为数组并取第0列
    # if len(set(dataList[:, -1].T.tolist()[0])) == 1:  # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分。
    #     #  exit cond 1
    #     return None, leafType(dataList)


    err = err_faction(dataList)
    # 赋初始值
    best_feat_idx, best_feat_val, best_err = 0, 0, float('inf')
    # 遍历所有特征
    # n-1
    for feat_idx in range(1, n):
        values = dataList[:, feat_idx]
        # 遍历所有特征值
        for val in values:
            # 按照当前特征和特征值分割数据
            left, right = split_data(dataList.tolist(), feat_idx, val)

            if len(left) < n_tolerance or len(right) < n_tolerance:
                # 如果切分的样本量太小
                continue

            # 计算误差
            new_err = err_faction(left) + err_faction(right)
            if new_err < best_err:
                best_feat_idx = feat_idx
                best_feat_val = val
                best_err = new_err

    # 如果误差变化并不大归为一类
    if abs(err - best_err) < err_tolerance:
        return None, leaf_faction(dataList)

    # 检查分割样本量是不是太小
    ldata, rdata = split_data(dataList.tolist(), best_feat_idx, best_feat_val)
    if len(ldata) < n_tolerance or len(rdata) < n_tolerance:
        return None, leaf_faction(dataList)

    return best_feat_idx, best_feat_val


def linear_regression(dataList):
    """
    获取线性回归系数
    因变量在第0列，其余为自变量
    :param dataList: 数据集
    :return: w: 回归系数，是一维矩阵
             X: 自变量矩阵
             y: 因变量矩阵
    """
    dataset = np.matrix(dataList)
    # 分割数据并添加常数列
    # X_ori, y = dataset[:, :-1], dataset[:, -1]
    X_ori, y = dataset[:, 1:], dataset[:, 0]
    X_ori, y = np.matrix(X_ori), np.matrix(y)
    # X_ori 少一列，y 只有一列
    m, n = X_ori.shape
    X = np.matrix(np.ones((m, n+1)))
    X[:, 1:] = X_ori

    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的不可逆，会造成程序异常
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of opt')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    w = xTx.I * (X.T * y)

    # print('线性回归：')
    # print(w)
    # print(X.shape)
    # print(X)
    # print(y.shape)
    # print(y)
    return w, X, y


def leaf_lmTree(dataList):
    """
    计算给定数据集的线性回归系数
    :param dataList: 数据集
    :return: 见 def linear_regression
    """
    w, _, _ = linear_regression(dataList)
    return w


def err_lmTree(dataList):
    """
    对给定数据集进行回归并计算误差
    :param dataList: 数据集
    :return: 见 def linear_regression
    """
    w, X, y = linear_regression(dataList)
    y_prime = X*w
    return np.var(y_prime - y)


