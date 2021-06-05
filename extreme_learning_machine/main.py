import numpy as np


def load_data():
    """
    加载本地数据
    :return:
    """
    train_x = []
    train_y = []
    fileIn = open('container_crane_controller.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))

    return np.mat(train_x), np.mat(train_y)


def g(tempH):
    """
    激活函数
    :param tempH:
    :return:
    """
    H = 1 / (1 + np.exp(-tempH))
    return H


train_x, train_y = load_data()

[N, n] = train_x.shape
L = 15

# 随机产生权重w和偏置b
W = np.random.rand(n, L)

b = np.random.rand(1, L)

# np.repeat 对行进行复制，如下：
# >>> a=np.array([[10,20],[30,40]])
#
# >>> a.repeat([3,2],axis=0)
#
# array([[10, 20],
#
#          [10, 20],
#
#          [10, 20],
#
#          [30, 40],
#
#          [30, 40]])
# 复制 N 次，从而与输入数据一一对应
b = np.repeat(b, N, axis=0)

tempH = train_x * W + b
H = g(tempH)
T = train_y.T
# m = 2
# T = np.zeros((N, m))
# for i in xrange(N):
#     if train_y[0, i] == 0:
#         T[i, 0] = 1
#     else:
#         T[i, 1] = 1
# 最小二乘求解
outputWeight = H.I * T

output = H * outputWeight
print('输出层权值')
print(outputWeight.T)
print('预测结果')
print(output.T)
print('真实结果')
print(train_y)
#print(np.sum((output > 0.5) == train_y.T) * 1.0 / N)

#print(np.sum(np.argmax(output, axis=1) == train_y.T) * 1.0 / N)