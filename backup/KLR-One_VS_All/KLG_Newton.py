# D:\Demo\PythonDemo
# -*- coding: utf-8 -*-
# @Time    : 2018/1/29 13:04
# @Author  : Tang
# @File    : KLG_Newton.py
# @Software: PyCharm
from __future__ import division
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

class KLR(object):
    def __init__(self, sigma=3, toler=1e-5, max_Iter=20000, penalty = 5, kernel_type='Gaussian'):
        self.kernels = {
            'Linear': self.kernel_linear,
            'Gaussian': self.kernel_gaussian
        }
        self.sigma = sigma
        self.toler = toler
        self.max_Iter = max_Iter
        self.penalty = penalty
        self.kernel_type = kernel_type
    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1 + np.exp(-Z))
    # Define kernels
    @staticmethod
    def kernel_linear(x1, x2):
        return np.dot(x1, x2.T)
    def kernel_gaussian(self,x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma**2))
    # Define kenelMatrix
    def kernelMat(self, X1, X2):  # 150*150的矩阵
        # sample size
        n1 = X1.shape[0]  # shape[0]查看行数
        n2 = X2.shape[0]
        kernel = self.kernels[self.kernel_type]
        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel(X1[i], X2[j])
        return K
    @staticmethod
    def initialized(y):
        alphas = [];m = len(y)
        m1 = 0;m2 = 0;C = 1
        for i in range(m):
            if y[i] == 1:
                m1 += 1
            else:
                m2 += 1
        for j in range(m):
            if y[j] == 1:
                alphas.append(C / m1)
            else:
                alphas.append(C / m2)
        return np.array(alphas)

    def Newton(self,X_train, y_train, k):
        loss = 0;conver = [];iter = 0
        m, n = np.shape(X_train)
        alpha = np.zeros(len(y_train))  # alpha 的初始值也很重要
        # k = self.kernelMat(X_train, X_train)  # 245X245
        # k = self.K(X_train)                             #近似核矩阵
        for j in range(self.max_Iter):
            loss_old = np.copy(loss)
            p = self.sigmoid(np.dot(k, alpha))  # 245X1
            v = np.diag([p[i] for i in range(len(p))])  # 245X245
            gradent = np.dot(k, y_train - p) - self.penalty * np.dot(k, alpha)  # 245X1
            gradent2 = np.linalg.multi_dot([k.T, v, k]) + self.penalty * k  # 245X245
            alpha = alpha + np.dot(np.linalg.pinv(gradent2), gradent)  # 求逆1X245
            loss = -1 / m * (
                np.dot(y_train, np.log(p)) + np.dot((1 - y_train), np.log(1 - p))) + self.penalty * np.linalg.multi_dot([alpha.T, k, alpha])
            c = np.linalg.norm(loss - loss_old)
            print(c)
            conver.append(c)
            iter += 1
            # print '迭代次数：', iter
            if c < 1e-5:
                break
        return alpha, conver
    def K(self, X):
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
        return K[:, :]


