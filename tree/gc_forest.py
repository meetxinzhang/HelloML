# forked from pylablanche/gcForest(https://github.com/pylablanche/gcForest)

import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# noinspection PyUnboundLocalVariable
class gcForest(object):

    def __init__(self, shape_1X=None, n_mgsRFtree=30, window=None, stride=1,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=np.inf,
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=1):
        """ gcForest Classifier.

        :param shape_1X: int or tuple list or np.array (default=None)
            Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!
            For sequence data a single int can be given.

        :param n_mgsRFtree: int (default=30)
            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)
            List of window sizes to use during Multi Grain Scanning.
            If 'None' no slicing will be done.

        :param stride: int (default=1)
            Step used when slicing the data.

        :param cascade_test_size: float or int (default=0.2)
            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)
            Number of Random Forests in a cascade layer.
            For each pseudo Random Forest a complete Random Forest is created, hence
            the total numbe of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)
            Number of trees in a single Random Forest in a cascade layer.

        :param min_samples_mgs: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Multi-Grain Scanning Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param min_samples_cascade: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Cascade Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param cascade_layer: int (default=np.inf)
            mMximum number of cascade layers allowed.
            Useful to limit the contruction of the cascade.

        :param tolerance: float (default=0.0)
            Accuracy tolerance for the casacade growth.
            If the improvement in accuracy is not better than the tolerance the construction is
            stopped.

        :param n_jobs: int (default=1)
            The number of jobs to run in parallel for any Random Forest fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        """
        setattr(self, 'shape_1X', shape_1X)
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)

        # 总森林个数 = 2 * n_cascadeRF
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        if isinstance(window, int):
            setattr(self, 'window', [window])
        elif isinstance(window, list):
            setattr(self, 'window', window)
        setattr(self, 'stride', stride)
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'cascade_layer', cascade_layer)
        setattr(self, 'min_samples_mgs', min_samples_mgs)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)
        setattr(self, 'n_jobs', n_jobs)

    def fit(self, X, y):
        """ Training the gcForest on input data X and associated target y.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array
            1D array containing the target values.
            Must be of shape [n_samples]
        """
        if np.shape(X)[0] != len(y):
            raise ValueError('Sizes of y and X do not match.')

        mgs_X = self.mg_scanning(X, y)
        _ = self.cascade_forest(mgs_X, y)

    def predict_proba(self, X):
        """ Predict the class probabilities of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class probabilities for each input sample.
        """
        mgs_X = self.mg_scanning(X)
        cascade_all_pred_prob = self.cascade_forest(mgs_X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)

        return predict_proba

    def predict(self, X):
        """ Predict the class of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        pred_proba = self.predict_proba(X=X)
        predictions = np.argmax(pred_proba, axis=1)

        return predictions

    def mg_scanning(self, X, y=None):
        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array
            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.
        """
        setattr(self, '_n_samples', np.shape(X)[0])
        shape_1X = getattr(self, 'shape_1X')
        # 判断 shape_1X 是否为 int 类型
        if isinstance(shape_1X, int):
            # 则输入为队列数据
            shape_1X = [1, shape_1X]
        if not getattr(self, 'window'):
            # 输入为图片数据
            setattr(self, 'window', [shape_1X[1]])

        mgs_pred_prob = []

        # 每个不同大小的窗口，都有两个森林：随机森林 和 完全随机森林
        for wdw_size in getattr(self, 'window'):
            # wdw_pred_prob 是随机森林和完全随机森林横向连接后的结果
            wdw_pred_prob = self.window_slicing_pred_prob(X, wdw_size, shape_1X, y=y)
            mgs_pred_prob.append(wdw_pred_prob)

            print("window size:{}, shape of out:{}".format(wdw_size, np.shape(wdw_pred_prob)))

        # 横向拼接不同窗口的 多粒度扫描 概率结果
        return np.concatenate(mgs_pred_prob, axis=1)

    def window_slicing_pred_prob(self, X, window, shape_1X, y=None):
        """ Performs a window slicing of the input data and send them through Random Forests.
        If target values 'y' are provided sliced data are then used to train the Random Forests.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample.

        :param y: np.array (default=None)
            Target values. If 'None' no training is done.

        :return: np.array
            Array of size [n_samples, ..] containing the Random Forest.
            prediction probability for each input sample.
        """
        n_tree = getattr(self, 'n_mgsRFtree')
        min_samples = getattr(self, 'min_samples_mgs')
        stride = getattr(self, 'stride')

        if shape_1X[0] > 1:
            print('Slicing Images...')
            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y, stride=stride)
        else:
            print('Slicing Sequence...')
            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y, stride=stride)

        # 训练
        if y is not None:
            n_jobs = getattr(self, 'n_jobs')
            prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
            crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
            print('Training MGS Random Forests...')
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
            # 保存模型参数
            setattr(self, '_mgsprf_{}'.format(window), prf)
            setattr(self, '_mgscrf_{}'.format(window), crf)
            pred_prob_prf = prf.oob_decision_function_
            pred_prob_crf = crf.oob_decision_function_

        # 测试
        if hasattr(self, '_mgsprf_{}'.format(window)) and y is None:
            # 复原模型参数
            prf = getattr(self, '_mgsprf_{}'.format(window))
            crf = getattr(self, '_mgscrf_{}'.format(window))
            pred_prob_prf = prf.predict_proba(sliced_X)
            pred_prob_crf = crf.predict_proba(sliced_X)

        # 连接随机森林和完全随机森林的结果
        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]

        # 转换形状，_n_samples 行，列数自适应
        return pred_prob.reshape([getattr(self, '_n_samples'), -1])

    def _window_slicing_img(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for images

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_cols].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced images and target values (empty if 'y' is None).
        """
        if any(s < window for s in shape_1X):
            raise ValueError('window must be smaller than both dimensions for an image')

        # 滑动时产生的窗口个数
        len_iter_x = np.floor_divide((shape_1X[1] - window), stride) + 1
        len_iter_y = np.floor_divide((shape_1X[0] - window), stride) + 1
        # 每个窗口的左上角坐标
        iterx_array = np.arange(0, stride*len_iter_x, stride)
        itery_array = np.arange(0, stride*len_iter_y, stride)

        ref_row = np.arange(0, window)

        # ref_ind 是图片中第一个窗口内的横坐标
        # np.ravel 降维，默认横向，i 表示行数
        ref_ind = np.ravel([ref_row + shape_1X[1] * i for i in range(window)])

        # 所有被窗口覆盖的点坐标
        # 关于 itertools.product()，求笛卡尔积，例：
        # list1 = ['a', 'b']
        # list2 = ['c', 'd']
        # print(itertools.product(list1, list2))
        # ('a', 'c')
        # ('a', 'd')
        # ('b', 'c')
        # ('b', 'd')
        inds_to_take = [ref_ind + ix + shape_1X[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        # reshape(-1, window**2) 表示转换成 window**2 列，行数自适应
        # sliced_imgs 每一行即是一个窗口
        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window**2)

        # 有标签，训练
        if y is not None:
            # 复制数组 y, 次数  len_iter_x * len_iter_y，即每个窗口对应一个标签
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        # 没标签，测试
        elif y is None:
            sliced_target = None

        return sliced_imgs, sliced_target

    def _window_slicing_sequence(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_col].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced sequences and target values (empty if 'y' is None).
        """
        if shape_1X[1] < window:
            raise ValueError('window must be smaller than the sequence dimension')

        len_iter = np.floor_divide((shape_1X[1] - window), stride) + 1
        # 每个窗口的第一个指标
        iter_array = np.arange(0, stride*len_iter, stride)

        # np.prod 元素相乘，这里一维的数据，得到的就是长度，用len()就可以了，这里作者在炫技啊
        ind_1X = np.arange(np.prod(shape_1X))
        # 被窗口覆盖的点坐标
        inds_to_take = [ind_1X[i:i+window] for i in iter_array]

        # np.take 安装第二个参数从第一个参数里取数据，例：
        # a = [[1, 2, 3],
        #      [4, 5, 6],
        #      [7, 8, 9]]
        # ids = [0, 2]
        #
        # c = np.take(a, ids)
        # d = np.take(a, ids, axis=1)
        # print(c, d)
        # c = [1, 3]
        # d = [[1, 3]
        #      [4, 6]
        #      [7, 9]]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        elif y is None:
            sliced_target = None

        return sliced_sqce, sliced_target

    def cascade_forest(self, X, y=None):
        """ Perform (or train if 'y' is not None) a cascade forest estimator.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        if y is not None:
            setattr(self, 'n_layer', 0)  # 这行代码跟构造方法里的重复了
            test_size = getattr(self, 'cascade_test_size')
            max_layers = getattr(self, 'cascade_layer')
            tol = getattr(self, 'tolerance')

            # sklearn.train_test_split 从样本中随机的按比例选取train data和testdata
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            # 初始值为0，这里是第一层
            self.n_layer += 1
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)
            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

            # 假如进行第二层，先评估一下
            self.n_layer += 1
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
            accuracy_layer = self._cascade_evaluation(X_test, y_test)

            # 如果下一层满足条件，一直循环添加新层
            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:
                # 更新层的准确率
                accuracy_ref = accuracy_layer
                # 更新层的输出
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                # 添加新层
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                accuracy_layer = self._cascade_evaluation(X_test, y_test)

            # 如果假如进行的第二层比第一层还差，那么便删掉这一层
            # 注意：
            #     accuracy_layer > accuracy_ref 但不满足 accuracy_layer > (accuracy_ref + tol) 时,
            #     第二层还是会保留，只是不会有第三层了。
            if accuracy_layer < accuracy_ref:
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                self.n_layer -= 1

        elif y is None:
            # at_layer 代表需要去调用的层，预测的时候从第一层开始
            at_layer = 1
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)
            # 一层一层向下遍历
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref

    def _cascade_layer(self, X, y=None, layer=0):
        """ Cascade layer containing Random Forest estimators.
        If y is not None the layer is trained.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :param layer: int (default=0)
            Layer indice. Used to call the previously trained layer.

        :return: list
            List containing the prediction probabilities for all samples.
        """
        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_samples_cascade')

        n_jobs = getattr(self, 'n_jobs')
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)

        prf_crf_pred = []
        if y is not None:
            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)

                # 获取预测概率，由于已经调用过 .fit()，所以不需要再使用.predict_proba() 了
                prf_crf_pred.append(prf.oob_decision_function_)
                prf_crf_pred.append(crf.oob_decision_function_)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                # list.append 是纵向连接对象（多个），不是拼接数据
                prf_crf_pred.append(prf.predict_proba(X))
                prf_crf_pred.append(crf.predict_proba(X))

        return prf_crf_pred

    def _cascade_evaluation(self, X_test, y_test):
        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array
            Array containing the test input samples.
            Must be of the same shape as training data.

        :param y_test: np.array
            Test target values.

        :return: float
            the cascade accuracy.
        """
        # 回调.cascade_forest，y为空，只调用预测函数，减小了代码重复性
        # np.mean(_, axis=0) 求每一列的均值
        # cascade_forest(X_test) 的 shape 为：[n_samples*2*n_rf, n_classes], 以n_samples 为一个单元
        # 因为在cascade_forest 中是直接.append 每个随机森林的输出
        # np.mean 后的 shape 为：[n_samples, n_classes]
        # 例： n_samples=5, n_rf=2
        # list = []
        # a = [[1, 2, 3],
        #      [4, 5, 6],
        #      [7, 8, 9],
        #      [1, 2, 3],
        #      [4, 5, 6]]
        # b = [[1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1]]
        #
        # list.append(a)
        # list.append(b)
        # list.append(a)
        # list.append(b)
        # print(list)
        # m = np.mean(list, axis=0)
        # print(m)
        #
        # [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
        #  [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        #  [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
        #  [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        #
        # [[1.  1.5 2. ]
        #  [2.5 3.  3.5]
        #  [4.  4.5 5.  ]
        #  [1.  1.5 2. ]
        #  [2.5 3.  3.5 ]]

        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)
        # 返回最大值的指标
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        # sklearn 的评估函数
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)
        print('Layer validation accuracy = {}'.format(casc_accuracy))

        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        """ Concatenate the original feature vector with the predicition probabilities
        of a cascade layer.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param prf_crf_pred: list
            Prediction probabilities by a cascade layer for X.

        :return: np.array
            Concatenation of X and the predicted probabilities.
            To be used for the next layer in a cascade forest.
        """
        # 关于 np.swapaxes 多维矩阵的转置
        # 例：
        # list = []
        # a = [[1, 2, 3],
        #      [4, 5, 6],
        #      [7, 8, 9],
        #      [1, 2, 3],
        #      [4, 5, 6]]
        # b = [[1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1],
        #      [1, 1, 1]]
        #
        # list.append(a)
        # list.append(b)
        # list.append(a)
        # list.append(b)
        # print(list)
        # sw = np.swapaxes(list, 0, 1)
        # print(sw)
        #
        # [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
        #  [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        #  [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]],
        #  [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        #
        # [[[1 2 3]
        #   [1 1 1]
        #   [1 2 3]
        #   [1 1 1]]
        #
        # [[4 5 6]
        #  [1 1 1]
        #  [4 5 6]
        #  [1 1 1]]
        #
        # [[7 8 9]
        #  [1 1 1]
        # [7 8 9]
        # [1 1 1]]
        #
        # [[1 2 3]
        #  [1 1 1]
        # [1 2 3]
        # [1 1 1]]
        #
        # [[4 5 6]
        #  [1 1 1]
        # [4 5 6]
        # [1 1 1]]]
        #
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        # 转换形状为：[n_samples, -1]
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        # 和 X 进行横向拼接
        feat_arr = np.concatenate([add_feat, X], axis=1)

        return feat_arr
