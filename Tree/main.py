from Tree.my_tree import MyTree
from Tree.r_forest import MyRandomForest
from Tree.tools import *

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 载入数据---------------------------------------------------------
    X_train, y_train, X_test = \
        load_list_data_and_standardization('train_data.txt',
                                           n_folds=10,
                                           idx_test=0,
                                           n_test=1)

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

    forest = MyRandomForest(tree_type='regression',
                            num_remove_feature=5,
                            opt={'err_tolerance': 1, 'n_tolerance': 20},
                            sample_ratio=0.7,
                            n_tree=100)
    # 训练
    mean_struct = forest.fit(X_train, y_train)
    print('森林中树结构的均值为：\n', mean_struct)
    # 测试
    yHats = forest.predict(X_test)
    print('预测结果为：\n', yHats)
    # 评估
    X_test = np.matrix(X_test)
    y_true = X_test[:, 0]
    mre(y_true, yHats)
    r2(y_true, yHats)

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
