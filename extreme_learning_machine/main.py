# Python standard libraries
import numpy as np
# My python files
from my_loader import iris, loan
from extreme_learning_machine.elm import ExtremeLearningMachine  # read elm.py for more details

# x, y = iris('Iris.csv')  #  https://www.kaggle.com/datasets/uciml/iris
x, y = loan('loan_data.csv')  # https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis
train_x = x[:9000]  # the first 9000 lines for training
train_y = y[:9000]
test_x = x[9000:]  # lines except first 9000 for test. Num of test should << num of training
test_y = y[9000:]
print('x.shape:', x.shape, len(train_x), 'for training and', len(test_x), 'for test.')  # [batch, 5]

# Training and test with a single-layer ELM demo.
elm = ExtremeLearningMachine(in_features=13,  # 13 columns of x
                             out_features=2,  # 2 categories
                             hidden_features=64)  # hyperparameter
elm.train(train_x, train_y)  # training  x:[batch, in_features=13], y:[batch,]
logits = elm.predict(test_x)  # prediction  [batch, out_features=2]

# Get the results
output_onehot = {0: [1, 0],
                 1: [0, 1]}
max_idx = np.argmax(logits, axis=1)  # find the index of max value  [batch, out_features=2] -> [batch, 1]
_y_onehot = [output_onehot[int(i)] for i in max_idx]  # convert index into one-hot -> [batch, 2]
print('\nPrediction')
print(_y_onehot)
print('Label')
print(np.array(test_y).tolist())

# Evaluation
_y_onehot = np.mat(_y_onehot)  # convert list into matrix
""" Example to calculate accuracy, if 3 categories
Prediction
[[0, 0, 1], [0, 0, 1], [0, 0, 1]
Label
[[0, 0, 1], [0, 1, 0], [1, 0, 0]
Hadamard product
[[0, 0, 1], [0, 0, 0], [0, 0, 0]
# Addition
[[1], [0], [0]]
# Accuracy
1/3=0.333333
"""
correct = np.array(np.multiply(_y_onehot, test_y))  # Hadamard product
correct = np.sum(correct, axis=1, keepdims=False)  # add all categories
print('Accuracy: ', sum(correct) / len(test_y))
