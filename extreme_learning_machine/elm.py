# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/12/9 20:42
 @desc:
"""
import numpy as np


def sigmoid(v):
    """activate function
    """
    return 1 / (1 + np.exp(-v))


class ExtremeLearningMachine:
    def __init__(self, in_features, out_features, hidden=64):
        self.f = in_features  # num of columns/attributes
        self.o = out_features
        self.h = hidden

        # hidden_layer: map features into another dimension, non-linear
        self.weight = np.random.rand(self.f, self.h)
        self.bias = np.random.rand(1, self.h)

        # output_layer: do prediction, linear
        self.beta = np.ones(shape=[self.h, self.o])

    def random_feature_mapping(self, x):
        [batch, in_features] = np.shape(x)
        assert in_features == self.f
        bias = np.repeat(self.bias, batch, axis=0)
        features = sigmoid(x * self.weight + bias)
        return features

    def train(self, x, y):
        features = self.random_feature_mapping(x)
        # Solve beta by Least Square
        self.beta = features.I * y.T

    def forward(self, x):
        features = self.random_feature_mapping(x)
        return features * self.beta
