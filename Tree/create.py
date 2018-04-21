from Tree.args import *
from Tree.predict import *
from PCA.p import *


def create_recursion_tree(dataList, leaf_faction, err_faction, opt=None):
    """
    创建递归树结构
    :return 树模型参数，字典类型
    """
    if opt is None:
        opt = {'err_tolerance': 1, 'n_tolerance': 4}

    # 选择最优化分特征和特征值
    feat_idx, value = choose_best_feature(dataList, leaf_faction, err_faction, opt)

    # 触底条件，此时 value 为回归系数矩阵，即方程参数模型
    if feat_idx is None:
        return value

    # 创建一层树结构
    tree = {'feat_idx': feat_idx, 'feat_val': value}

    # 递归创建左子树和右子树
    ldata, rdata = split_data(dataList, feat_idx, value)
    ltree = create_recursion_tree(ldata, leaf_faction, err_faction, opt)
    rtree = create_recursion_tree(rdata, leaf_faction, err_faction, opt)
    tree['left'] = ltree
    tree['right'] = rtree

    return tree


def load_list_data(filename):
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

    print(type(dataList))
    print(np.shape(dataList))
    return dataList
