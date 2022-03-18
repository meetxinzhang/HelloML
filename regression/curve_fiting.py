import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# x = [1107.107, 1145.549, 1658.206, 2122.415, 2167.438, 2268.839]
# y = [57.25959, 94.20663, 340.3132, 183.2433, 98.09858, 74.7973]
# x = [1653.458496, 1727.418254, 2342.227051, 2511.978923, 2596.681478, 2651.28776]
# y = [62.53567631, 167.9751948, 204.2364543, 95.35409838, 71.85523901, 72.73040719]
x = [575.9510905, 700.3417765, 2697.336426, 5041.006348, 5280.247559]
y = [96.03100345, 264.9302531, 1174.717723, 926.5340403, 875.0267348]


properties = np.polyfit(x, y, 3)
function = np.poly1d(properties)

xvals = range(0, int(np.max(x))+10, 10)
yvals = function(xvals)

y_max = np.max(yvals)

x_plot = []
y_plot = []
print('x: ', x)
print('y: ', y)
print('      x                  y')
print('var ', np.var(x), np.var(y))
print('std ', np.std(x), np.std(y))

print('\nfitting func: \n', function)
for x_ in xvals:
    y_ = function(x_)
    if y_ == y_max:
        print('\nThe max Y and relative X: ', y_max, x_)
    if y_ >= 0:
        x_plot.append(x_)
        y_plot.append(y_)

# plot
# plt.scatter(x, y)
plot1 = plt.plot(x, y, '*', label='original values')
plot2 = plt.plot(x_plot, y_plot, 'r', label='polyfit values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()
