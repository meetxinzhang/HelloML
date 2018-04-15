from Tree.args_tree import *
from Tree.predict_tree import *
from sklearn.decomposition import PCA


def create_tree(dataList, leafType, errType, opt=None):
    """
    创建递归树结构
    :return 树模型参数，字典类型
    """
    if opt is None:
        opt = {'err_tolerance': 1, 'n_tolerance': 4}

    # 选择最优化分特征和特征值
    feat_idx, value = choose_best_feature(dataList, leafType, errType, opt)

    # 触底条件，此时 value 为回归系数矩阵，即方程参数模型
    if feat_idx is None:
        return value

    # 创建一层树结构
    tree = {'feat_idx': feat_idx, 'feat_val': value}

    # 递归创建左子树和右子树
    ldata, rdata = split_data(dataList, feat_idx, value)
    ltree = create_tree(ldata, leafType, errType, opt)
    rtree = create_tree(rdata, leafType, errType, opt)
    tree['left'] = ltree
    tree['right'] = rtree

    return tree


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



train_data = load_data('train_data.txt')
test_data = load_data('test_data.txt')

pca = PCA(n_components=15)

tree = create_tree(train_data, leaf_faction, err_faction, opt={'err_tolerance': 1, 'n_tolerance': 901})
print(tree)

print('1111111111111111111111111111111111111111111111')
yHat = createForeCast(tree, test_data, modelEval=modelTreeEval)
print(yHat)

