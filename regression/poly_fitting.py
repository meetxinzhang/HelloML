# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: poly_fitting.py
@time: 6/13/21 7:15 PM
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt


def min_max_normalization(arr):
    return [float(x - np.min(arr)) / (np.max(arr) - np.min(arr)) for x in arr]


def mean_normaliztion(arr):
    return [float(x - arr.mean()) / arr.std() for x in arr]


def sigmoid(arr):
    return 1. / (1 + np.exp(-arr))


# x = [575.95109, 700.34178, 2697.33643, 5041.00635, 5280.24756]
# y = [96.031, 264.93025, 1174.71772, 926.53404, 875.02673]
# x = [57.25959, 94.20663, 340.31325, 183.24334, 98.09858]
# y = [1107.10706, 1145.54858, 1658.20587, 2122.41504, 2167.43831]
x = [1653.4585, 1727.41825, 2342.22705, 2511.97892, 2596.68148]
y = [62.53568, 223.25964, 204.23645, 95.3541, 71.85524]

print('x: ', x)
print('y: ', y)
print('  var', '              std')
print('x', np.var(x), np.std(x))
print('y', np.var(y), np.std(y))

# x = min_max_normalization(x)
# y = min_max_normalization(y)

plt.scatter(x, y)

properties = np.polyfit(x, y, 3)  # 用3次多项式拟合
func = np.poly1d(properties)
print('\nfitting func: \n', func)  # 在屏幕上打印拟合多项式

xvals = range(0, int(np.max(x)) + 10, 10)
yvals = func(xvals)
y_max = np.max(yvals)

x_plot = []
y_plot = []

for _x, _y in zip(xvals, yvals):
    if _y == y_max:
        print('\nThe max y and relative x: ', _y, _x)
    if _y >= 0:
        x_plot.append(_x)
        y_plot.append(_y)

# plt.plot(x, y, '*', label='original values')
plt.plot(x_plot, y_plot, 'r', label='fitting values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()
