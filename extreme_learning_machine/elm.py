# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/12/9 20:42
 @desc:

 ELM[1] was proposed for single-hidden layer feedforward neural networks (SLFNs) which randomly chooses hidden nodes and
 analytically determines the output weights of SLFNs. In theory, this algorithm tends to provide good generalization
 performance at extremely fast learning speed.

 ELM include two layers, named hidden-layer and output-layer, I prefer to call them `random-features-mapping-layer` and
 `learnable-layer`, since new names indicating their functions.

 Ref.
  [1] Huang, G et al,.  Extreme learning machine: theory and applications. Neurocomputing. 2006. 70(1/3), 489-501.

"""
import numpy as np


def sigmoid(v):
    """activation function, to offer the non-linear-mapping
    """
    return 1 / (1 + np.exp(-v))


class ExtremeLearningMachine:
    def __init__(self, in_features, out_features=1, hidden_features=64):
        self.f = in_features  # num of columns/attributes
        self.o = out_features  # num of output dimension, default=1
        self.h = hidden_features  # num of hidden layer, default=64
        self.param_c = 1
        self.param_opt = True

        # random-features-mapping-layer, non-linear
        self.weight = np.random.rand(self.f, self.h)  # extend the dimension of input into h by matrix multiply.
        self.bias = np.random.rand(1, self.h)  # bias added with each sample.

        # learnable-layer: do prediction, linear, learnable.
        self.beta = np.ones(shape=[self.h, self.o])

    def random_feature_mapping(self, x):
        [batch, in_features] = np.shape(x)
        assert in_features == self.f
        bias = np.repeat(self.bias, batch, axis=0)  # copy bias, [1,h] -> [batch,h]
        features = sigmoid(x * self.weight + bias)  # [batch,f] * [f,h] + [batch,h] -> [batch,h]
        return features

    def train(self, x, y):
        """Solve beta
        Define output=features*beat, loss=||output-y||^2
        To minimize the loss, the output should close to y, that is: y~=features*beat
        The idea of ELM is to let beat=features^-1*y, than you will see:
              features*beta = features*features^-1*y = I*y = y
        So, the key is to solve the inverse matrix of features (Moore-Penrose generalized inverse matrix).
          Note: Moore-Penrose generalized inverse ensure the unique inverse exists for any m*n matrix.
        """
        # Random-features-mapping-layer, or named hidden-layer in original paper.
        features = self.random_feature_mapping(x)  # [batch,f] -> [batch,h]

        """
        Easier way to find inverse by NumPy:
            inverse = features.I  # find the inverse
        So the beta is:
            self.beta = inverse * y  # [h, batch] * [batch,o] -> [h,o]
         
        But I want to solve it by manual this time.
        There are two ways:
            1) Solve directly the linear matrix equation: beta=(HTH)^-1*HT*y -> HTH*beta=I*HT*y
                         beta = np.linalg.solve(HTH, I*HT*y)
            2) follows the formulation of Moore-Penrose generalized inverse
                         the inverse of H = (HTH)^-1*HT
        """
        HT = features.T  # [h,batch]
        # if self.param_opt:
        #     self.beta = np.linalg.solve(
        #         (np.eye(HT.shape[0]) / self.param_c) + HT * HT.conj().T,
        #         HT * y)
        # else:
        #     self.beta = HT * np.linalg.solve(
        #         ((np.eye(HT.shape[1]) / self.param_c) + HT.conj().T * HT), y)

        HTH = HT * features  # [h,batch]*[batch,h] -> [h,h]
        if np.linalg.det(HTH) == 0:
            p = np.linalg.pinv(HTH)
        else:
            p = np.linalg.inv(HTH)
        inverse = p * HT
        # inverse = features.I  # find the inverse
        # Update beta
        self.beta = inverse * y  # beta = (HTH)^-1 * HT * y

    def predict(self, x):
        features = self.random_feature_mapping(x)
        return features * self.beta  # [batch,h] * [h,o] -> [batch,o]
