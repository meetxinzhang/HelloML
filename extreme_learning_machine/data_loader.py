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


class IterForever:
    """An exception captured extension for next(iter(iterable)) to make the next operator recurrent.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.it = iter(zip(x, y))

    def get_len(self):
        return self.x.shape[0]

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            del self.it
            self.it = iter(zip(self.x, self.y))
            return next(self.it)


def make_batch(x, y, batch_size=16):
    """ divide x and y into many batch, e.g. x=[n,c]->[n/batch_size, batch_size, c]
    """
    dropout = len(x) % batch_size
    if dropout == 0:
        x = np.array(x)
        y = np.array(y)
    else:
        x = np.array(x[:-dropout])
        y = np.array(y[:-dropout])
    c_x = x.shape[1]
    c_y = y.shape[1]
    x = x.reshape((-1, batch_size, c_x))
    y = y.reshape((-1, batch_size, c_y))
    return IterForever(x, y)


def normalization(x):
    # normalization
    _range = np.max(x) - np.min(x)
    x = (x-np.min(x)) / _range
    # standardization
    mean = np.mean(x)
    std = np.std(x)
    return (x-mean) / std


def divide_train_test(x, y, divide=0.7, batch_size=None):
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
    if batch_size == None:
        return np.mat(train_x), np.mat(train_y), np.mat(test_x), np.mat(test_y)  # One-shot training version
    else:
        return make_batch(train_x, train_y, batch_size), make_batch(test_x, test_y, batch_size)


def loan(path, divide=0.7, batch_size=None):
    """ load Lending Club Loan Dataset
    Args:
        path: full local path
        divide: the rate of train_x/x, default=0.7
        batch_size: int, set it None to use one-shot training
    """
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
    # x = normalization(x)
    return divide_train_test(x, y, divide, batch_size)


def iris(path, divide=0.7, batch_size=None):
    """ load Iris Dataset
    Args:
        path: full local path
        divide: the rate of train_x/x, default=0.7
        batch_size: int, set it None to use one-shot training
    """
    x = []
    y = []
    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i == 0: continue  # skip the first row (columns name)
            x.append([float(e) for e in row[:-1]])  # x, float, except the last column
            y.append(iris_onehot[row[-1]])  # y, last column
    # x = normalization(x)
    return divide_train_test(x, y, divide, batch_size)
