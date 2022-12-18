# Numpy is available
import numpy as np
# My python files
from data_loader import iris, loan
from extreme_learning_machine.elm import ExtremeLearningMachine  # read elm.py for more details


def evaluation(logits, y, classes=2):
    batch = len(y)
    _y_onehot = np.array([[0]*classes]*batch)
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
    # Iris dataset https://www.kaggle.com/datasets/uciml/iris
    # train_x, train_y, test_x, test_y = iris('Iris.csv')
    # output_onehot = {0: [1, 0, 0],
    #                  1: [0, 1, 0],
    #                  2: [0, 0, 1]}

    # loan dataset https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis
    train_x, train_y, test_x, test_y = loan('loan_data.csv', divide=0.7)
    output_onehot = {0: [1, 0],
                     1: [0, 1]}

    # Training and test with a single-layer ELM demo.
    elm = ExtremeLearningMachine(in_features=13,  # Columns of x. Loan: 13  Iris: 5
                                 out_features=2,  # Categories.  Loan: 2   Iris: 3
                                 hidden_features=128)  # hyperparameter
    elm.train(train_x, train_y)  # training  x:[batch, in_features=13], y:[batch,]
    logits = elm.predict(test_x)  # prediction  [batch, out_features=2]

    # Evaluation
    accuracy = evaluation(logits, test_y, classes=2)
    print('Accuracy: ', accuracy)
