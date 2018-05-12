from sklearn import metrics
# MRE 平均相对误差
# R^2 决定系数


def mre(y_true, y_pre):
    score = metrics.mean_absolute_error(y_true, y_pre, multioutput='uniform_average')
    print('平均绝对误差为：\n', score)
    pass


def r2(y_true, y_pre):
    score = metrics.r2_score(y_true, y_pre, multioutput='uniform_average')
    print('决定系数 R^2 为：\n', score)
    pass




