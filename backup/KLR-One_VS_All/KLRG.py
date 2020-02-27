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

class KLRG(object):
    def __init__(self,C=1,sigma=3,toler=1e-5,max_Iter=10000,kernel_type='Gaussian'):
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
            # print("diff:", diff)
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
        FMat = np.dot((alphas*y).T, KernelMat) # 带入核矩阵计算alpha*y = 105X1
        HMat = FMat + (np.log(delte / (1 - delte)) * y)
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

    def K(self, X):
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
        return K[:, :]