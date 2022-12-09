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


x, y = load_data()
[r, c] = x.shape
print('x.shape: ', r, c)
h = 15


# from extreme_learning_machine.elm import ExtremeLearningMachine
# elm = ExtremeLearningMachine(in_features=2, out_features=1, hidden=15)
# elm.train(x, y)
# _y = elm.forward(x)
# print(_y.T)


def sigmoid(v):
    """激活函数
    """
    return 1 / (1 + np.exp(-v))


# initiate the weight and bias by randomness.
W = np.random.rand(c, h)
b = np.random.rand(1, h)

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
b = np.repeat(b, r, axis=0)

# map the x into high-dimensional space.
hid = sigmoid(x * W + b)
yT = y.T

# m = 2
# T = np.zeros((N, m))
# for i in xrange(N):
#     if train_y[0, i] == 0:
#         T[i, 0] = 1
#     else:
#         T[i, 1] = 1

# 最小二乘求解
outputWeight = hid.I * yT
output = hid * outputWeight

print('输出层权值')
print(outputWeight.T)
print('预测结果')
print(output.T)
print('真实结果')
print(y)

# print(np.sum((output > 0.5) == train_y.T) * 1.0 / N)
# print(np.sum(np.argmax(output, axis=1) == train_y.T) * 1.0 / N)

#
# a = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# a = np.mat(a)
# print(a.I)
#
# aT = np.transpose(a)
# aTa = np.dot(aT, a)
#
# p = np.linalg.inv(aTa)
# paT = np.dot(p, aT)
# print(paT)
