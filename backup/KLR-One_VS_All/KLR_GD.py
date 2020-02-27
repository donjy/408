# D:\Demo\PythonDemo
# -*- coding: utf-8 -*-
# @Time    : 2018/1/29 11:54
# @Author  : Tang
# @File    : KLR_GD.py
# @Software: PyCharm
from __future__ import division
import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
filepath = os.path.dirname(os.path.abspath(__file__))
class KLR(object):
    def __init__(self, sigma=3,tep = 0.001, toler=1e-5, max_Iter=20000, penalty = 0.5,kernel_type='Gaussian'):
        self.kernels = {
            'Linear': self.kernel_linear,
            'Gaussian': self.kernel_gaussian
        }
        self.sigma = sigma
        self.tep = tep
        self.toler = toler
        self.max_Iter = max_Iter
        self.penalty = penalty
        self.kernel_type = kernel_type
    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1 + np.exp(-Z))
    # Define kernels3
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

    #初始化alpha
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
    def SGD(self,k,X_train, y_train):
        m, n = np.shape(X_train)
        alpha = np.zeros(len(y_train))  # alpha 的初始值也很重要
        yk = - (y_train * k)  # 分子
        loss = 0;conver =[];ite,err = 0,0
        for j in range(self.max_Iter):
            loss_old = np.copy(loss)
            yalphak = np.dot((alpha * y_train).T, k).reshape(1, -1)
            denom = 1 + np.exp(yalphak)  # 求分母
            for i in range(m):
                err = np.sum(yk[i] / denom, axis=0)
            gradent = err + self.penalty * np.dot(k, alpha)
            alpha = alpha - self.tep * gradent
            denom2 = 1 + np.exp(-yalphak)
            loss = 1 / m * np.sum(np.log(denom2)) + self.penalty * np.linalg.multi_dot([alpha.T, k, alpha])
            c = np.linalg.norm(loss - loss_old)
            conver.append(c)
            ite += 1
            # print '迭代次数：', ite
            if c < self.toler:
                break
        return alpha, conver
    def K(self, X):
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
        return K[:, :]



