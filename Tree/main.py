import numpy as np
from Tree.create import load_list_data, create_recursion_tree
from Tree.args import leaf_lmTree, err_lmTree
from Tree.predict import predict_test_data, predict_lmTree
from Tree.r_forest import random_forest, random_forest_predict
from PCA.main import outPut


# 载入数据----------------------------------------------------
# 训练数据 16维
print("训练数据：")
train_data = load_list_data('train_data.txt')
# 检验缺省值
print(np.isnan(train_data).any())
# 测试数据 15维
print("测试数据：")
test_data = load_list_data('test_data.txt')
# 检验缺失值
print(np.isnan(test_data).any())

# 随机森林


# 影响预测准确性，弃用
# # 预处理：使用 PCA 将数据变为线性无关-------------------------------------
# # 分割出自变量，因为PCA只处理自变量
# train_data = np.matrix(train_data)
# temp_X = train_data[:, 1:]
#
# pca_X = outPut(temp_X, 15)
#
# # 复原数据
# train_data[:, 1:] = pca_X
# train_data = train_data.tolist()


# 树> logistic model tree --------------------------------------------
# tree = create_recursion_tree(train_data,
#                              leaf_faction=leaf_lmTree,
#                              err_faction=err_lmTree,
#                              opt={'err_tolerance': 1, 'n_tolerance': 901})
# print('树结构为：')
# print(tree)
#
# # 使用树预测
# print('预测结果为：')
# yHat = predict_test_data(tree,
#                          test_data,
#                          predictFacion=predict_lmTree)
# print(yHat)


# 随机森林-----------------------------------------------------------
forest = random_forest(train_data, ratio=0.3, n_tree=10)
print('森林结构为：')
print(forest)

print('预测结果为：')
yHats = random_forest_predict(forest, test_data)
print(yHats)
