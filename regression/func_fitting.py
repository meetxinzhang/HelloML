# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: func_fitting.py
@time: 6/15/21 9:15 PM
@desc:
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# x = [575.95109, 700.34178, 2697.33643, 5041.00635, 5280.24756]
# y = [96.031, 264.93025, 1174.71772, 926.53404, 875.02673]
# x = [57.25959, 94.20663, 340.31325, 183.24334, 98.09858]
# y = [1107.10706, 1145.54858, 1658.20587, 2122.41504, 2167.43831]
x = [1653.4585, 1727.41825, 2342.22705, 2511.97892, 2596.68148]
y = [62.53568, 223.25964, 204.23645, 95.3541, 71.85524]

print('x: ', x)
print('y: ', y)
# print('  var', '              std')
# print('x', np.var(x), np.std(x))
# print('y', np.var(y), np.std(y))


def function(x, i, n):
    return i*x - (x**2)/n


props, err = curve_fit(function, x, y)

x_contiguous = range(0, int(np.max(x)) + 10, 10)
y_contiguous = [function(_x, props[0], props[1]) for _x in x_contiguous]

print('\ny='+str(props[0])+' * X - X^2 / '+str(props[1]))
plt.scatter(x, y)
# plt.plot(x, y, '*', label='original values')
plt.plot(x_contiguous, y_contiguous, 'r', label='fitting curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()
