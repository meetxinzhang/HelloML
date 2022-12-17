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

loan_purpose = {'debt_consolidation': 0,
                'credit_card': 1,
                'major_purchase': 2,
                'home_improvement': 3,
                'small_business': 4,
                'educational': 5,
                'all_other': 6}

loan_onehot = {'0': [1, 0],
               '1': [0, 1]}


def loan(path):
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
    return np.mat(x), np.mat(y)


# Define a dictionary to map string name in csv.
iris_onehot = {'Iris-setosa': [1, 0, 0],
               'Iris-versicolor': [0, 1, 0],
               'Iris-virginica': [0, 0, 1]}


def iris(path):
    x = []
    y = []
    with open(path) as file:
        for i, row in enumerate(csv.reader(file)):
            if i == 0: continue  # skip the first row (columns name)
            x.append([float(e) for e in row[:-1]])  # x, float, except the last column
            y.append(iris_onehot[row[-1]])  # y, last column
    return np.mat(x), np.mat(y)
