import numpy as np
import csv


# def load_txt(path):
#     x = []
#     y = []
#     with open(path) as file:
#         for line in file.readlines():
#             line = line.strip().split(',')
#             x.append([float(line[0]), float(line[1])])
#             y.append(float(line[2]))
#     return np.mat(x), np.mat(y).T


iris_dic = {'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2}


def load_csv(path):
    x = []
    y = []
    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i==0: continue
            x.append([float(e) for e in row[:-1]])
            y.append(iris_dic[row[-1]])
    return np.mat(x), np.mat(y).T


x, y = load_csv('Iris.csv')
# x, y = load_txt('container_crane_controller.txt')
print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

from extreme_learning_machine.elm import ExtremeLearningMachine
elm = ExtremeLearningMachine(in_features=5, out_features=1, hidden_features=32)
elm.train(x, y)
_y = elm.predict(x)


print('\nPrediction')
print(_y.T)
print('Label')
print(y.T)
