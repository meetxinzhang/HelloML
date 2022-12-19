# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/12/15 20:33
 @desc:
"""
import numpy as np
# Python basic library
import csv

# Define a dictionary to map the first column of loan dataset.
loan_purpose = {'debt_consolidation': 0,
                'credit_card': 1,
                'major_purchase': 2,
                'home_improvement': 3,
                'small_business': 4,
                'educational': 5,
                'all_other': 6}

# 2 classes in loan dataset
loan_onehot = {'0': [1, 0],
               '1': [0, 1]}

# 3 classes in Iris dataset
iris_onehot = {'Iris-setosa':     [1, 0, 0],
               'Iris-versicolor': [0, 1, 0],
               'Iris-virginica':  [0, 0, 1]}


def divide_train_test(x, y, divide=0.7):
    # Shuffle the x and y at the same time
    data = list(zip(x, y))
    np.random.shuffle(data)
    x[:], y[:] = zip(*data)

    # divide into training, test sets
    train_num = int(len(x) * divide)
    train_x = x[:train_num]  # the first 9000 lines for training
    train_y = y[:train_num]
    test_x = x[train_num:]  # lines except first 9000 for test. Num of test should << num of training
    test_y = y[train_num:]
    print('dataset size:', len(x), ', ', len(train_x), 'for training and', len(test_x), 'for test.')  # [batch, 5]
    return np.mat(train_x), np.mat(train_y), np.mat(test_x), np.mat(test_y)


def loan(path, divide=0.7):
    x = []
    y = []

    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i == 0: continue  # skip the first row (columns name)
            policy = int(row[0])
            purpose = loan_purpose[row[1]]
            a = [float(e) for e in row[2:-1]]
            x.append([policy, purpose] + a)
            y.append(loan_onehot[row[-1]])  # y, last column
    return divide_train_test(x, y, divide)


def iris(path, divide=0.7):
    x = []
    y = []
    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i == 0: continue  # skip the first row (columns name)
            x.append([float(e) for e in row[:-1]])  # x, float, except the last column
            y.append(iris_onehot[row[-1]])  # y, last column
    return divide_train_test(x, y, divide)
