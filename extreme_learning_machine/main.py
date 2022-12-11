# Python standard libraries
import numpy as np
import csv

# Define the dictionary from string name to one-hot.
iris_onehot = {'Iris-setosa':     [1, 0, 0],
               'Iris-versicolor': [0, 1, 0],
               'Iris-virginica':  [0, 0, 1]}
# Define the dictionary from num to one-hot.
output_onehot = {0: [1, 0, 0],
                 1: [0, 1, 0],
                 2: [0, 0, 1]}


def load_csv(path):
    x = []
    y = []
    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i == 0: continue  # skip the first row (columns name)
            x.append([float(e) for e in row[:-1]])  # x, float, except the last column
            y.append(iris_onehot[row[-1]])  # y, last column
    return np.mat(x), np.mat(y)


x, y = load_csv('Iris.csv')  # Iris Species dataset, https://www.kaggle.com/datasets/uciml/iris
train_x = x[:125]  # the first 125 are for training
train_y = y[:125]
test_x = x[125:]  # the last 25 are for test. Num of test should << num of training since the
test_y = y[125:]
print('x.shape:', x.shape, len(train_x), 'for training and', len(test_x), 'for test.')  # [batch, 5]

from extreme_learning_machine.elm import ExtremeLearningMachine  # read elm.py for more details
elm = ExtremeLearningMachine(in_features=5,  # 5 columns of x
                             out_features=3,  # 3 categories
                             hidden_features=64)
elm.train(train_x, train_y)  # training
logits = elm.predict(test_x)  # prediction

# get the results
max_idx = np.argmax(logits, axis=1)  # find the index of max value for each sample
_y_onehot = [output_onehot[int(i)] for i in max_idx]  # convert index into one-hot
print('\nPrediction')
print(_y_onehot)
print('Label')
print(np.array(test_y).tolist())

# Evaluation
_y_onehot = np.mat(_y_onehot)
""" example to read following codes
Prediction
[[0, 0, 1], [0, 0, 1], [0, 0, 1]
Label
[[0, 0, 1], [0, 1, 0], [1, 0, 0]
Hadamard product
[[0, 0, 1], [0, 0, 0], [0, 0, 0]
# Addition
[1, 0, 0]
# Accuracy
1/3=0.333333
"""
correct = np.array(np.multiply(_y_onehot, test_y))  # Hadamard product
correct = np.sum(correct, axis=1, keepdims=False)  # add all categories
print('Accuracy: ', sum(correct) / len(test_y))
