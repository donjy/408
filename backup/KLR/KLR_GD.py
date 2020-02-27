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

class KLR(object):
    def __init__(self, sigma=3,tep = 0.001, toler=1e-5, max_Iter=50000, penalty = 0.5,kernel_type='Gaussian'):
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
    # 核矩阵近似求解方法
    # def K(self, X):
    #     from mklaren.kernel.kinterface import Kinterface
    #     from mklaren.kernel.kernel import rbf_kernel
    #     K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
    #     from mklaren.projection.nystrom import Nystrom
    #     model = Nystrom(rank=25)
    #     model.fit(K)
    #     G_nyst = model.G
    #     K_prox = np.dot(G_nyst, G_nyst.T)
    #     return K_prox
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
                err = np.sum(yk[i] / denom, axis=0)   ######
            gradent = err + self.penalty * np.dot(k, alpha)
            alpha = alpha - self.tep * gradent
            denom2 = 1 + np.exp(-yalphak)
            loss = 1 / m * np.sum(np.log(denom2)) + self.penalty * np.linalg.multi_dot([alpha.T, k, alpha])
            print(loss)
            c = np.linalg.norm(loss - loss_old)
            conver.append(c)
            ite += 1
            # print '迭代次数：', ite
            if c < self.toler:
                break
        return alpha, conver
    def Accuracy(self,X_train, X_test, y_train, y_test):
        start = time.time()
        k = self.kernelMat(X_train, X_train)
        # k = self.K(X_train)
        alphas, conver = self.SGD(k,X_train, y_train)

        end = time.time()
        TrainTime = (end - start)
        z = np.dot(self.kernelMat(X_test, X_train), alphas)
        y_proba = self.sigmoid(z)
        m = len(y_proba)
        TP, TN = 0, 0
        for i in range(m):
            if y_proba[i] > 0.5 and y_test[i] == 1:
                TP += 1
            elif y_proba[i] <= 0.5 and y_test[i] == -1:
                TN += 1
        accuracy = (TP + TN) / m
        print "Accuracy:" ,accuracy
        print "TP: %d , TN: %d" %(TP,TN)
        print "训练时间：" ,TrainTime
        # plt.plot(conver[1:-1])
        # plt.show()
        # return accuracy,TrainTime
#读取数据集
def loadFile(fileName):
    with open(fileName,'rb') as DataSet:
        df = pd.read_csv(DataSet,header=None)
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]
        return X.values , Y.values
X,y = loadFile(r'E:\DataSets\ionosphere.csv')

# def loadFile(filename, header=True):
#     data, header = [], None
#     with open(filename, 'rb') as csvfile:
#         spamreader = csv.reader(csvfile, delimiter=',')
#         if header:
#             header = spamreader.next()
#         for row in spamreader:
#             data.append(row)
#     return (np.array(data), np.array(header))
# (data, _) = loadFile(r'D:\DataSet\iris-slwc.txt',header=False)
# data = data.astype(float)
# X, y = data[:,0:-1], data[:,-1].astype(int)


#X_train:245X34     y_train:245L    X_test:106X34   y_test:106L
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = KLR()
    model.Accuracy(X_train, X_test, y_train, y_test)
    # from sklearn.model_selection import KFold
    # kf = KFold(n_splits=10, random_state=0, shuffle=True)
    # acc,GramT, TrainT,max_acc, min_acc = 0,0,0,0, np.inf
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #     accuracy, GramTime, TrainTime = model.Accuracy(X_train, X_test, y_train, y_test)
    #     if accuracy > max_acc:
    #         max_quota = accuracy
    #     elif accuracy < min_acc:
    #         min_quota = accuracy
    #     acc += accuracy / 10
    #     GramT += GramTime / 10
    #     TrainT += TrainTime / 10
    # print "Accuracy：" ,acc
    # print "GramTime:" ,GramT
    # print "TrainTime:",TrainT


