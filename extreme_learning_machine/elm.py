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


def invert_by_svd(matrix):
    """Solve Moore-Penrose generalized/pseudo inverse by Singular value decomposition.
    Args:
        matrix: [b,h]
    """
    matrix = matrix.conjugate()
    u, s, vt = np.linalg.svd(matrix, full_matrices=False, hermitian=False)  # [b,b] [h] [h,h]

    # [r, c] = matrix.shape
    # if r != c:  # if matrix is not a square
    #     assert r > c
    #     adjoint = np.zeros(shape=[c, (r - c)])
    #     inv_d = np.concatenate((inv_d, adjoint), axis=1)

    threshold = np.asarray([1e-10]) * np.max(s, keepdims=True)
    mask = s > threshold
    inv_s = np.divide(1, s, where=mask)  # [h]
    inv_s[~mask] = 0  # ~ inverse operation, True<->False

    inverse = np.matmul(vt.T,
                        np.multiply(np.expand_dims(inv_s, axis=-1),  # Hadamard product with broadcast
                                    u.T))    # [h,h]*[h,b]*[b,b]
    return inverse


class ExtremeLearningMachine:
    def __init__(self, in_features, out_features=1, hidden_features=64, method='svd'):
        self.i = in_features  # num of columns/attributes
        self.o = out_features  # num of output dimension, default=1
        self.h = hidden_features  # num of hidden layer, default=64
        self.norm_rate = 1
        self.method = method

        # random-features-mapping-layer, non-linear
        self.weight = np.random.rand(self.i, self.h)  # extend the dimension of input into h by matrix multiply.
        self.bias = np.random.rand(1, self.h)  # bias added with each sample.

        # learnable-layer: do prediction, linear, learnable.
        self.beta = np.ones(shape=[self.h, self.o])  # weight for output
        self.inv_ftf = None  # [h,h], temp value during training

    def random_feature_mapping(self, x):
        [batch, in_features] = np.shape(x)
        assert in_features == self.i
        bias = np.repeat(self.bias, batch, axis=0)  # copy bias, [1,h] -> [batch,h]
        features = sigmoid(np.matmul(x, self.weight) + bias)  # [batch,f] * [f,h] + [batch,h] -> [batch,h]
        return features

    def train(self, x, y):
        """Solve beta
        Define output=features*beat, loss=||output-y||^2
        To minimize the loss, the output should close to y, that is: y~=features*beat
        The idea of ELM is to let beat=features^-1*y, than you will see:
              features*beta = features*features^-1*y = I*y = y
        So, the key is to solve the inverse matrix of features (Moore-Penrose generalized/pseudo inverse matrix).
          Note: Moore-Penrose generalized/pseudo inverse ensure the unique inverse exists for any m*n matrix.
        Args:
            x: [batch, in_features] abbreviate [b, i]
            y: [batch, 1] abbreviate [b, 1]
        """
        # Random-features-mapping-layer, or named hidden-layer in original paper.
        features = self.random_feature_mapping(x)  # [b=batch,f] -> [b,h]
        """
        Easier way to find Moore-Penrose generalized inverse by NumPy:
            np.linalg.pinv = features.I or np.linalg.pinv(features)
        So the beta is:
            self.beta = inverse * y  # [h, batch] * [batch,o] -> [h,o]
        But I want to solve it by manual this time!!!!!!!!!!
        There are two ways:
            1) The Orthogonal project method, when features is non-singular
                         generalized inverse = (ftf)^-1*ft
            2) Singular value decomposition (SVD), can be generally used.
        """
        ft = features.T  # [h,b]
        ftf = np.matmul(ft, features)  # [h,b]*[b,h] -> [h,h], use for the Orthogonal project.
        if np.linalg.det(ftf) != 0:  # non-singular
            print('Non-singular matrix, use 1) The Orthogonal project.')
            # norm = np.eye(ftf.shape[0]) * self.norm_rate
            # ftf = np.multiply(ftf, norm)  # Normalization here, but weak effect.
            self.inv_ftf = np.linalg.inv(ftf)  # Compute the (multiplicative) inverse.
            inverse = np.matmul(self.inv_ftf, ft)  # [h,h]*[h,b] -> [h,b], (ftf)^-1 * ft -> f^-1 * ft^-1 * ft -> f^-1
        else:  # singular, the inverse does not exist
            print('Singular matrix, use 2) Singular_value_decomposition.')
            inverse = invert_by_svd(features)
            self.inv_ftf = invert_by_svd(ftf)  # Just for online-training
            # print('2.2 Solve linear matrix equation of orthogonal project: singular matrix')
            #  because, beta = (ftf) ^ -1 * ft * y -> ftf * beta = I * ft * y
            #  so, beta = np.linalg.solve(ftf, I * ft * y)
            # self.beta = np.linalg.solve(
            #     (np.eye(ft.shape[0]) / self.norm_rate) + np.matmul(ft, ft.conj().T),
            #     np.matmul(ft, y))
            # self.inv_ftf = singular_v_decomposition(ftf)  # Just for online-training

        # Update beta
        self.beta = np.matmul(inverse, y)  # beta = (ftf)^-1 * ft * y

    def online_train(self, x, y):
        """OS-ELM
        Liang N Y, Huang G B, Saratchandran P, et al. A fast and accurate online sequential learning algorithm for
        feedforward networks[J]. IEEE Transactions on neural networks, 2006, 17(6): 1411-1423.
        !!!!!!! Too many formulations
        """
        b = x.shape[0]  # batch
        features = self.random_feature_mapping(x)

        ft = features.T  # [h,b]
        eye = np.eye(b)  # [b,b]
        inv_ft = np.matmul(features, self.inv_ftf)  # [b,h]*[h,h]->[b,h], f*(ftf)^-1->f*f^-1*ft^-1->ft^-1->(f^-1).T
        leak_eye = np.matmul(inv_ft, ft)  # [b,h] * [h,b] -> [b,b]
        temp = np.linalg.inv(eye + leak_eye)  # [b,b]
        inv_f = np.matmul(self.inv_ftf, ft)  # [h,h]*[h,b] -> [h,b], (ftf)^-1*ft -> f^-1
        self.inv_ftf -= np.matmul(np.matmul(inv_f, temp), inv_ft)  # [h,h]-[h,b]*[b,b]*[b,h] -> [h,h]

        inverse = np.matmul(self.inv_ftf, ft)  # []
        fea_beta = np.matmul(features, self.beta)
        self.beta += np.matmul(inverse, (y - fea_beta))

    def predict(self, x):
        features = self.random_feature_mapping(x)
        return np.matmul(features, self.beta)  # [batch,h] * [h,o] -> [batch,o]
