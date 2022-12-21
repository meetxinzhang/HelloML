# Numpy is available
import numpy as np
# Python basic library
import time
# My python files
from data_loader import iris, loan
from extreme_learning_machine.elm import ExtremeLearningMachine  # read elm.py for more details


def evaluation(logits, y, classes=2):
    batch = len(y)
    _y_onehot = np.array([[0] * classes] * batch)
    max_idxes = np.argmax(logits, axis=1)  # find the index of max value  [batch, out_features=2] -> [batch, 1]
    for i, idx in enumerate(max_idxes):
        _y_onehot[i][idx] = 1
    print('\nPrediction')
    print(_y_onehot.tolist())
    print('Label')
    print(np.array(y).tolist())

    """ Example to calculate accuracy, if 3 classes
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
    return sum(correct) / batch


if __name__ == '__main__':
    # # Iris dataset https://www.kaggle.com/datasets/uciml/iris
    # train_x, train_y, test_x, test_y = iris('Iris.csv', divide=0.8, batch_size=None)  # One-shot training version
    # train_loader, test_loader = iris('Iris.csv', divide=0.8, batch_size=4)  # train batch-by-batch
    # in_features = 5; classes = 3

    # loan dataset https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis
    train_x, train_y, test_x, test_y = loan('loan.csv', divide=0.1, batch_size=None)  # One-shot training version
    # train_loader, test_loader = loan('loan.csv', divide=0.7, batch_size=16)  # train batch-by-batch
    in_features = 13; classes = 2

    # Training and test with a single-layer ELM demo.
    elm = ExtremeLearningMachine(in_features=in_features,  # Columns of x. Loan: 13  Iris: 5
                                 out_features=classes,  # Categories.  Loan: 2   Iris: 3
                                 hidden_features=128)  # hyperparameter

    # # One-shot training version
    T1 = time.time()
    elm.train(train_x, train_y)
    T2 = time.time()
    print('\nTraining takes ', ((T2 - T1) * 1000), ' ms')
    logits = elm.predict(test_x)  # prediction  [batch, out_features=2]
    acc = evaluation(logits, test_y, classes=classes)
    print('Accuracy: ', acc)

    # # Train batch-by-batch version
    # accuracy = []
    # # Initial training
    # (train_x, train_y) = train_loader.next()
    # elm.train(train_x, train_y)
    #
    # # Online training
    # n = train_loader.get_len()
    # for step in range(n - 1):
    #     (train_x, train_y) = train_loader.next()
    #     elm.online_train(train_x, train_y)  # training  x:[batch, in_features=13], y:[batch,]
    #
    #     if step % 1 == 0:
    #         (test_x, test_y) = test_loader.next()
    #         logits = elm.predict(test_x)  # prediction  [batch, out_features=2]
    #         acc = evaluation(logits, test_y, classes=classes)
    #         accuracy.append(acc)
    #         print('Accuracy: ', acc)
    #
    # """visualization by matplotlib
    # $ pip install matplotlib, install the library if need
    # """
    # # import matplotlib.pyplot as plt
    # # step = range(n-1)
    # # plt.plot(step, accuracy, label='Accuracy', linewidth=1)
    # # plt.show()
