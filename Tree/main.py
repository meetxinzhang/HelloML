from Tree.my_tree import MyTree
from Tree.r_forest import MyRandomForest
from Tree.tools import *

from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # 载入数据---------------------------------------------------------
    X_train, y_train, X_test, y_test = \
        load_list_data_and_standardization('all.txt',  # 文件名
                                           n_folds=5,  # 把数据分成n_folds份
                                           idx_test=0,  # 测试数据从第几份开始
                                           n_test=1)  # 测试数据所占的份数
    # X_train = usePCA(X_train)
    # X_test = usePCA(X_test)

    #  ---------------单独使用 logistic model tree 进行训练预测----------
    # tree = MyTree(tree_type='regression',
    #               num_remove_feature=0,
    #               opt={'err_tolerance': 1, 'n_tolerance': 4})
    # # 训练
    # struct = tree.fit(X_train, y_train)
    # print('树结构为：\n', struct)
    #
    # # 预测
    # yHat = tree.predict(X_test)
    # print('预测结果为：\n', yHats)

    # --------------------使用随机森林进行训练预测------------------------

    forest = MyRandomForest(tree_type='regression',  # 树参数：树类型，暂时只支持LMT，用作回归
                            num_remove_feature=5,  # 树参数：构建树时，随机去掉的特征数量
                            opt={'err_tolerance': 5, 'n_tolerance': 100},
                            # 树参数：预剪枝用到，'err_tolerance': 左右子树最小允许误差，'n_tolerance'：左右子树最小允许样本数
                            sample_ratio=0.7,  # 随机森林参数：构建树的时候随机抽样所占总样本的比例
                            n_tree=50)  # 随机森林参数：树的数量
    forest.fit(X_train, y_train)
    yHats = forest.predict(X_test)

    # # sklearn
    sk_forest = RandomForestRegressor(n_estimators=50, min_samples_split=100, n_jobs=10,  max_features='sqrt')
    sk_forest.fit(X_train, y_train)
    sk_yHats = forest.predict(X_test)
    print('sk-learn 结果：')
    mre(y_test, sk_yHats)
    r2(y_test, sk_yHats)

    print('\n 结果对比： real \t predict \n')
    for i in zip(y_test, yHats):
        print(''.join('%s \t' % s for s in i))

    # 评估
    mre(y_test, yHats)
    r2(y_test, yHats)

    # --------------------使用深度森林进行训练预测------------------------
    # iris = load_list_data('iris.txt')
    # iris = np.matrix(iris)
    # y_iris = iris[:, -1]
    # X_iris = np.delete(iris, -1, axis=1)
    # X_iris = np.array(X_iris)
    # y_iris = np.ravel(np.array(y_iris))
    #
    # X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.33)
    #
    # print('train_data: ', np.shape(X_tr))
    # print('test_data: ', np.shape(X_te))

    # from sklearn.datasets import load_iris, load_digits
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)
    # # print(np.shape(X_tr))
    # # print(np.shape(y_tr))
    # # print(y_tr)
    #
    # gcf = gcForest(shape_1X=[1, 4], window=2, tolerance=0.0)
    # gcf.fit(X_tr, y_tr)
    # y_pre = gcf.predict(X_te)
    # print('预测值为：', y_pre)
    # print('真实值为：', y_te)
    # print('acc: ', accuracy_score(y_te, y_pre, normalize=True, sample_weight=None))
    pass
