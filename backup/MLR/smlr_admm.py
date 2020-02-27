# coding=utf-8
import time
from operator import itemgetter

from numpy.linalg import norm
from scipy.sparse import csc_matrix
from sklearn import metrics
import os
import pandas as pd
import numpy as np
from numpy import *
from scipy import optimize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
    1.将Admm算法对象化
    2.求解过程全部换为矩阵形式
'''

def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

# def SoftMax(a, kappa):
#     return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)

def SoftMax(kappa, a):
    # return np.sign(a)*max(np.abs(x)-kappa,0)
    a_ = a.copy()
    a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
    a_[np.where(a_ > kappa)] -= kappa
    a_[np.where(a_ < - kappa)] += kappa
    return a_

def label_validate(y):
    if y.min() > 0:
        y -= 1
    return y

class SparseSoftmaxAdmm(object):

    def __init__(self, rho, Iter, lamda):
        self.rho = rho
        self.admmIter = Iter
        self.lamda = lamda

    def calculate_loss(self, X, y, theta, u_z):
        m = X.shape[0]
        margin = X.dot(theta)  # Guess that in some case, it might be "predictions = X.dot(self.W) + b"
        margin -= margin.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        softmax = np.e ** margin
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        # loss_obj = -safe_log(softmax[np.arange(len(softmax)), y]).sum() / m
        # regu_obj = norm(theta + u_z) ** 2
        # tot_obj = loss_obj + self.rho / 2.0 * regu_obj
        softmax[np.arange(len(softmax)), y] -= 1
        dW = X.T.dot(softmax) / m + self.rho * (theta + u_z)
        return dW

    def update_x_adam(self, config, X, y, theta, u_z):
        items = itemgetter('learning_rate', 'eps', 'beta_1', 'beta_2')(config)
        learning_rate, eps, beta_1, beta_2 = items
        config['t'] = 0
        config['m'] = np.zeros(theta.shape)
        config['v'] = np.zeros(theta.shape)
        # config.setdefault('t', 0)
        # config.setdefault('m', np.zeros(theta.shape))
        # config.setdefault('v', np.zeros(theta.shape))

        for it in xrange(0, self.admmIter):
            gra = self.calculate_loss(X, y, theta, u_z)

            config['t'] += 1
            config['m'] = config['m']*beta_1 + (1-beta_1)*gra
            config['v'] = config['v']*beta_2 + (1-beta_2)*gra**2
            m = config['m']/(1-beta_1**config['t'])
            v = config['v']/(1-beta_2**config['t'])
            theta = theta - learning_rate*m / (np.sqrt(v)+eps)
            # print str(it) + " " + str(list(theta[0]))
        return theta

    def update_x_admm(self, X, y, x, u_z, rho):
        (m, n) = X.shape

        def func(theta, *args):
            X, y = args
            theta = theta.reshape(u_z.shape)
            margin = X.dot(theta)  # Guess that in some case, it might be "predictions = X.dot(self.W) + b"
            margin -= margin.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
            softmax = np.e ** margin
            softmax /= softmax.sum(axis=1).reshape([-1, 1])
            loss_obj = -safe_log(softmax[np.arange(len(softmax)), y]).sum() / m
            regu_obj = norm(theta + u_z) ** 2
            tot_obj = loss_obj + rho / 2.0 * regu_obj
            softmax[np.arange(len(softmax)), y] -= 1
            dW = X.T.dot(softmax) / m + rho * (theta + u_z)
            return tot_obj, dW

        optionFlag = {'maxiter': 50, 'disp': False}
        res = optimize.minimize(func, x, method='L-BFGS-B', jac=True, options=optionFlag, args=(X, y))
        return res.x

    def train(self, X, y):
        labels = set(y)
        numLabel = len(labels)
        x = zeros((X.shape[1], numLabel))
        z = zeros((X.shape[1], numLabel))
        u = zeros((X.shape[1], numLabel))
        iter = 0
        while iter < self.admmIter:

            config = {'learning_rate': 0.01, 'eps': 1e-8, 'beta_1': 0.9, 'beta_2': 0.999}
            x = self.update_x_adam(config, X, y, x, u-z)
            # x = self.update_x_admm(X, y, x, u-z, self.rho)
            # print list(x[0, :])

            # for i, label in enumerate(labels):
            #     z[:, i] = SoftMax(self.lamda / self.rho, (x[:, i] + u[:, i]))
            z = SoftMax(self.lamda / self.rho, (x + u))

            u = x + u - z

            iter += 1
            # print "iter: " + str(iter) + " " + str(list(x[:, 0]))
            print "iter: {}, test_obj: {}".format(iter, self.test_obj(X, y, x))
            print "iter: {}".format(iter)
        return x, labels

    def test_obj(self, X, y, theta):
        m = X.shape[0]
        margin = X.dot(theta)  # Guess that in some case, it might be "predictions = X.dot(self.W) + b"
        margin -= margin.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        softmax = np.e ** margin
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        loss_obj = -safe_log(softmax[np.arange(len(softmax)), y]).sum() / m
        regu_obj = np.sum(np.abs(theta))
        tot_obj = loss_obj + self.rho / 2.0 * regu_obj
        return tot_obj

    """模型预测"""
    def predict(self, X, theta):
        margin = X.dot(theta)
        margin -= margin.max(axis=1).reshape([-1, 1])
        softmax = np.e ** margin
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        preds = softmax.argmax(axis=1).astype(float)
        return preds

    """模型评价"""
    def evaluate(self, preds, trues):
        hit = 0
        for pred, true in zip(preds, trues):
            if pred == true:
                hit += 1
        p = 1. * hit / trues.shape[0]
        print "accuracy: " + str(p)
        return p

    """获取(X, y)"""
    def get_label_points(self, data, load_sparse=False):
        # y = data['label'].astype(int)
        # X = data.drop(['label'], axis=1)
        # return X.values, y.values

        tmp = data.values
        y = tmp[:, -1].astype(int)
        X = tmp[:, :-1]

        if load_sparse: X = csc_matrix(X)

        return X, y

    """按比例划分训练集测试集进行训练与预测"""
    def random_split_fit(self, train_path, test_size=0.33, seed=1, load_sparse=False):
        dtrain = pd.read_csv(train_path)
        X, y = self.get_label_points(dtrain, load_sparse)  ; y = label_validate(y)
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=test_size, random_state=seed)
        theta, labels = self.train(xTrain, yTrain)
        preds = self.predict(xTest, theta)
        self.evaluate(preds, yTest)

    """分别提供训练集、测试集做训练与预测"""
    def train_test_fit(self, train_path, test_path, load_sparse=False):
        dtrain = pd.read_csv(train_path)
        dtest = pd.read_csv(test_path)
        xTrain, yTrain = self.get_label_points(dtrain, load_sparse)  ; yTrain = label_validate(yTrain)
        xTest, yTest = self.get_label_points(dtest, load_sparse)     ; yTest = label_validate(yTest)

        scaler = StandardScaler()
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest  = scaler.transform(xTest)

        theta, labels = self.train(xTrain, yTrain)
        preds = self.predict(xTest, theta)
        self.evaluate(preds, yTest)

    """K折交叉验证"""
    def kFold_fit(self, train_path, n_folds=5, load_sparse=False, random_state=1):
        dtrain = pd.read_csv(train_path)
        X, y = self.get_label_points(dtrain, load_sparse)            ; y = label_validate(y)
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
        accuracies = []
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print "Fold", i
            xTrain = X[train_index]
            yTrain = y[train_index]
            xTest = X[test_index]
            yTest = y[test_index]
            theta, labels = self.train(xTrain, yTrain)
            preds = self.predict(xTest, theta)
            accuracies.append(self.evaluate(preds, yTest))
            print "acc: " + str(metrics.accuracy_score(yTest, preds))
        print "avg: " + str(mean(accuracies))

if __name__ == '__main__':
    start = time.clock()

    # TODO: get_label_points()
    base = 'F:\\MyDocuments\\2016-09-Hyperspectral\\Classification\\'
    (train_path, test_path) = None, None
    # train_path = base + 'MNIST.csv'
    # train_path = base + 'GTcropped32_32.csv'
    train_path = base + 'COIL20.csv'
    # train_path = base + 'ORL_head.csv'
    # train_path = base + 'segment-challenge.csv'
    # test_path = base + 'segment-test.csv'
    # train_path = base + 'waveform-+noise-train.csv'
    # test_path = base + 'waveform-+noise-test.csv'
    # train_path = base + 'mnist_train_zscore_sk.csv'
    # test_path = base + 'mnist_test_zscore_sk.csv'
    # train_path = base + 'stl10_train.csv'
    # test_path = base + 'stl10_test.csv'
    # train_path = base + 'YLB_96_84_rm_last.csv' # precision: 0.95608531995
    # train_path = base + 'YLB_96_84_train_sk.csv'
    # test_path = base + 'YLB_96_84_test_sk.csv'
    # train_path = os.path.join(base, 'cifar-10-batches-py', 'cifar-train.csv')
    # test_path = os.path.join(base, 'cifar-10-batches-py', 'cifar-test.csv')

    s = SparseSoftmaxAdmm(rho=1e-4, Iter=50, lamda=1e-4)
    # s.train_test_fit(train_path, test_path, load_sparse=False)
    s.random_split_fit(train_path, test_size=0.33, seed=1)
    # s.kFold_fit(train_path, n_folds=3, random_state=3)

    end = time.clock()
    print "running: {} min, {} s".format(((end - start)/60.0), end - start)