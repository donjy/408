# D:\Demo\PythonDemo
# -*- coding: utf-8 -*-
# @Time    : 2017/12/2 12:51
# @Author  : Tang
# @File    : KLRG.py
# @Software: PyCharm
from __future__ import division
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import time
#filepath = os.path.dirname(os.path.abspath(__file__))   #引入当前文件夹中的数据集
class KLRG(object):
    def __init__(self,C=1,sigma=2,toler=1e-5,max_Iter=20000,kernel_type='Gaussian'):
        self.kernels ={
            'Linear' : self.kernel_linear,
            'Gaussian':self.kernel_gaussian
        }
        self.C = C
        self.sigma = sigma
        self.toler = toler
        self.max_Iter = max_Iter
        self.kernel_type = kernel_type
    #计算分类
    def accuracy(self,X_train,X_test,y_train,y_test):
        start = time.time()
        # KernelMat = self.kernelMat(X_train,X_train)
        # KernelMat = self.K(X_train)
        KernelMat = self.MKL(X_train,y_train)
        # KernelMat = self.MKL_prox(X_train,y_train)
        alphas ,b= self.calcNewAlpha(X_train, y_train,KernelMat)
        end = time.time()
        TrainTime=(end - start)
        # all_theta = np.dot((alphas * y_train), X_train).reshape(1, -1)
        # all_theta = all_theta.reshape(-1,1)
        # z = np.dot(X_test, all_theta) + b
        z = np.dot((alphas * y_train), self.kernelMat(X_train, X_test)) - b
        y_proba = self.sigmoid(z)
        m = len(y_proba)
        TP, TN = 0, 0
        for i in range(m):
            if y_proba[i] > 0.5 and y_test[i] == 1:
                    TP += 1
            elif y_proba[i] <= 0.5 and y_test[i] == -1:
                    TN += 1
        accuracy = (TP + TN) / m
        return accuracy, TrainTime ,TP ,TN
    #更新alpha，计算b
    def calcNewAlpha(self,X, y,KernelMat):
        iter = 0
        alphas = self.initialized(y)
        while (iter < self.max_Iter):
            alphas_old = np.copy(alphas)
            bup, blow, i, j = self.selectij(y, alphas,KernelMat)
            t = self.calcT(X, y, alphas, i, j,KernelMat)
            alphaiT, alphajT = self.alphaT(alphas, i, j, t,y)
            alphas[i] = alphaiT
            alphas[j] = alphajT
            iter += 1
            diff = np.linalg.norm(alphas - alphas_old)
            # print "alphas更新位置:", i, j
            if diff < self.toler or blow >= bup - 2 * self.toler:  # i,j不再迭代时,alpha也不变。
                b = (bup + blow) / 2
                return alphas, b
    #初始化alpha参数
    def initialized(self,y):
        alphas = [] ;m = len(y)
        m1 = 0;m2 = 0
        for i in range(m):
            if y[i] == 1:
                m1 += 1
            else:
                m2 += 1
        for j in range(m):
            if y[j] == 1:
                alphas.append(self.C / m1)
            else:
                alphas.append(self.C / m2)
        return np.array(alphas)
    #选择更新的alpha坐标
    def selectij(self, y, alphas, KernelMat):
        delte = alphas / self.C  # delte向量
        FMat = np.dot((alphas*y).T, KernelMat).reshape(-1,1) # 带入核矩阵计算alpha*y = 105X1
        HMat = FMat + (np.log(delte / (1 - delte)) * y).reshape(-1, 1)
        bup,blow = np.max(HMat), np.min(HMat)
        i, j = HMat.argmax(), HMat.argmin()
        return bup, blow, i, j
    #计算更新规则参数t
    def calcT(self,X, y, alphas, i, j ,KernelMat):
        iter = 0;   t = 0
        kernel = self.kernels[self.kernel_type]
        eta = kernel(X[i], X[i]) - 2 * kernel(X[i], X[j]) + kernel(X[j], X[j])
        while (iter < self.max_Iter):
            alphaiT, alphajT = self.alphaT(alphas, i, j, t,y)
            alphat = np.copy(alphas)    #计算alphat向量
            alphat[i] = alphaiT
            alphat[j] = alphajT
            deltet = alphat / self.C
            FMat = np.dot((alphat * y).reshape(1, -1), KernelMat.T)[0,0]
            HMat = FMat + (np.log(deltet / (1 - deltet)) * y)
            Hi = HMat[i] ; Hj = HMat[j]
            G2 = 1 / (deltet * (1 - deltet))
            G2i = G2[i] ; G2j = G2[j]
            phi2 = eta + (G2i + G2j) / self.C
            phi1 = Hi - Hj
            iter += 1
            t_old = t
            t = t - (phi1 / phi2)
            if (phi1 < 0.1 * self.toler) or (np.linalg.norm(t_old - t)) < self.toler:
                return t
    #计算更新的alphat
    @staticmethod
    def alphaT(alphas, i, j, t,y):  # 带入训练集
        alphaiT = alphas[i] + t / y[i]
        alphajT = alphas[j] - t / y[j]
        return alphaiT, alphajT
    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1 + np.exp(-Z))
    #Define kernels
    @staticmethod
    def kernel_linear(x1,x2):
        return np.dot(x1,x2.T)
    def kernel_gaussian(self,x1,x2):
        return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (self.sigma**2)))
    #Define kenelMatrix
    def kernelMat(self,X1, X2):  # 150*150的矩阵
        # sample size
        n1 = X1.shape[0]  # shape[0]查看行数
        n2 = X2.shape[0]
        kernel = self.kernels[self.kernel_type]
        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel(X1[i], X2[j])
        print(K.shape)
        return K
    # 核矩阵近似求解方法
    def K(self,X):
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
        from mklaren.projection.nystrom import Nystrom
        model = Nystrom(rank=15)
        model.fit(K)
        G_nyst = model.G
        K_prox = np.dot(G_nyst,G_nyst.T)
        #原始核矩阵
        # k_init = self.kernelMat(X,X)
        # print "the RMSE of prox_kernel:", np.var(K_prox-k_init)**0.5
        return K_prox

    def MKL(self, X_train, y_train):
        from mklaren.kernel.kernel import linear_kernel, poly_kernel
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K_exp = Kinterface(data=X_train, kernel=rbf_kernel, kernel_args={"sigma": 2})  # RBF kernel
        # K_exp2 = Kinterface(data=X_train, kernel=rbf_kernel, kernel_args={"sigma": 3})  # RBF kernel
        K_poly = Kinterface(data=X_train, kernel=poly_kernel, kernel_args={"degree": 3})  # polynomial kernel with degree=3
        K_lin = Kinterface(data=X_train, kernel=linear_kernel)  # linear kernel
        from mklaren.mkl.alignf import Alignf
        model2 = Alignf(typ="convex")
        model2.fit([K_exp, K_lin, K_poly], y_train)
        mu = model2.mu  # kernel weights
        mul_K = mu[0] * K_exp[:,:] + mu[1] * K_lin[:,:] + mu[2] * K_poly[:,:]
        k_init = self.K(X_train)
        print "the RMSE of prox_kernel:", np.var(mul_K - k_init) ** 0.5
        print "the weight of MKL:", mu
        return mul_K

    def MKL_prox(self, X_train, y_train):
        from mklaren.kernel.kernel import linear_kernel, poly_kernel
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K_exp = Kinterface(data=X_train, kernel=rbf_kernel, kernel_args={"sigma": 2})  # RBF kernel
        K_poly = Kinterface(data=X_train, kernel=poly_kernel, kernel_args={"degree": 3})  # polynomial kernel with degree=3
        K_lin = Kinterface(data=X_train, kernel=linear_kernel)  # linear kernel
        from mklaren.mkl.mklaren import Mklaren
        #使用最小角回归进行多核学习的同时执行低秩近似
        model2 = Mklaren(rank=25, lbd=0, delta=30)
        model2.fit([K_exp, K_lin, K_poly], y_train)
        #近似的核矩阵
        G_exp = model2.data[0]["G"]
        exp_approx = G_exp.dot(G_exp.T)

        G_lin = model2.data[1]["G"]
        lin_approx = G_lin.dot(G_lin.T)

        G_poly = model2.data[2]["G"]
        poly_approx = G_poly.dot(G_poly.T)

        mu = model2.mu  # kernel weights
        print mu
        K_approx = mu[0] * exp_approx + mu[1] * lin_approx + mu[2] * poly_approx

        k_init = self.kernelMat(X_train, X_train)
        print "the RMSE of prox_kernel:", np.var(K_approx - k_init) ** 0.5

        return K_approx



def loadFile(fileName):
    with open(fileName,'rb') as DataSet:
        df = pd.read_csv(DataSet,header=None)
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]
        return X.values , Y.values
X,y = loadFile(r'E:\DataSets\ionosphere.csv')


# df = pd.read_csv(r'E:\DataSets\MNIST_head.csv')
# df = df.notnull
# df_filer_lable = df[(df["label"] == 1) | (df["label"] == 7)]
# df_filer_lable["lable"] = df_filer_lable["label"].apply(lambda line: -1 if line==7 else 1)
# X = df_filer_lable.iloc[:, :-1]
# Y = df_filer_lable.iloc[:, -1]
# X,y = X.values, Y.values


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = KLRG()
    accuracy, TrainTime ,TP ,TN = model.accuracy(X_train,X_test,y_train,y_test)
    print "TP和TN：%d %d" %(TP,TN)
    print "accuracy:",accuracy
    print "TrainTime:",TrainTime



    #
    # from sklearn.model_selection import KFold
    # # print "ACC:" , accuracy
    # kf = KFold(n_splits=5, random_state=0, shuffle=True)
    # acc, TrainT,max_acc, min_acc = 0,0,0, np.inf
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    #     accuracy, TrainTime, TP, TN = model.accuracy(X_train, X_test, y_train, y_test)
    #     if accuracy > max_acc:
    #         max_quota = accuracy
    #     elif accuracy < min_acc:
    #         min_quota = accuracy
    #     acc += accuracy / 5
    #     TrainT += TrainTime / 5
    # print "Accuracy：" ,acc
    # print "TrainTime:",TrainT