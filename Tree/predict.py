from numpy import *


def predict_lmTree(model, inDat):
    """
    预测，一行测试数据 * 回归系数
    对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
    也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
    :param model -- 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型，实则为 回归系数
    :param inDat -- 输入的测试数据
    :return float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    # print(shape(X))
    # print(shape(model))

    return float(X * model)


def recursion_tree(tree, inData, tree_type='regression'):
    """
    迭代遍历树
    :param tree -- 已经训练好的树的模型
    :param inData -- 输入的测试数据，只有一行
    :param tree_type -- 树类型
    :return 返回预测值
    """
    if tree_type == 'regression':
        predict_faction = predict_lmTree
    else:
        predict_faction = predict_lmTree
        # TODO

    if not isTree(tree):
        return predict_faction(tree, inData)
    # 书中写的是inData[tree['spInd']]，只适合inData只有一列的情况，否则会产生异常
    if inData[0, tree['feat_idx']] <= tree['feat_val']:
        # 可以把if-else去掉，只留if里面的分支
        if isTree(tree['left']):
            return recursion_tree(tree['left'], inData, tree_type)
        else:
            return predict_faction(tree['left'], inData)
    else:
        # 同上，可以把if-else去掉，只留if里面的分支
        if isTree(tree['right']):
            return recursion_tree(tree['right'], inData, tree_type)
        else:
            return predict_faction(tree['right'], inData)


def predict_test_data(tree, testData, tree_type):
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
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    # print yHat
    for i in range(m):
        yHat[i, 0] = recursion_tree(tree, mat(testData[i]), tree_type)
        # print "yHat==>", yHat[i, 0]
    return yHat


# 判断节点是否是一个字典
def isTree(obj):
    """
    测试输入变量是否是一棵树,即是否是一个字典
    :param
        obj -- 输入变量
    :return
        返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
    """
    return type(obj).__name__ == 'dict'



