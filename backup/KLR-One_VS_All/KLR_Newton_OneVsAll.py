#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.io as spio
from KLG_Newton import KLR

class OneVsAll(object):
    def __init__(self, num_labels=16, test_size=0.2, sigma=3):
        self.num_labels = num_labels
        self.test_size = test_size
        self.sigma = sigma

    @staticmethod
    def loadmat_data(fileName):
        return spio.loadmat(fileName)
    @staticmethod
    def sigmoid(z):
        h = np.zeros((len(z), 1))  # 初始化，与z的长度一致
        h = 1.0 / (1.0 + np.exp(-z))
        return h
    def kernel_gaussian(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.sigma ** 2)))
    def kernelMat(self, X1, X2):  # 150*150的矩阵
        # sample size
        n1 = X1.shape[0]  # shape[0]查看行数
        n2 = X2.shape[0]
        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_gaussian(X1[i], X2[j])
        return K

    #划分数据集
    def train_test_split(self,X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        return X_train, X_test, y_train, y_test
    #映射训练集的label为-1/1
    def map_label(self, y_train):
        num_y = len(y_train)
        class_y = np.zeros((num_y, self.num_labels)) - 1  # 训练数据的y对应0-9，需要映射为-1/1的关系
        for i in range(self.num_labels):
            class_y[:, i] = np.int32(y_train == i).reshape(1, -1)  # 注意reshape(1,-1)才可以赋值
        # y_train = np.array([list(map(lambda e: -1 if e == 0 else 1, i)) for i in class_y])
        return class_y

    #计算所有标签类别里的all_z
    def calc_z(self,KLRG_model,X_train,X_test,y_train,KernelMat):
        m, n = X_test.shape
        kernelMat = self.kernelMat(X_test, X_train)
        all_Zs = np.zeros((m, self.num_labels))  # 初始化每一列对应相应分类的z
        for i in range(self.num_labels):
            print("训练第{}个分类器".format(i))
            alphas, conver = KLRG_model.Newton(X_train, y_train[:, i], KernelMat)
            z = np.dot(kernelMat, alphas)
            all_Zs[:, i] = z.reshape(1, -1)  # 放入all_theta中
        return all_Zs
    #计算预测结果
    def calc_acc(self,all_z, X_test, y_test):
        m, n = X_test.shape
        h = self.sigmoid(all_z)
        '''
        返回h中每一行最大值所在的列号
        - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
        - 最后where找到的最大概率所在的列号（列号即是对应的数字）
        '''
        p = np.array(np.where(h[0, :] == np.max(h, axis=1)[0]))
        for i in np.arange(1, m):
            t = np.array(np.where(h[i, :] == np.max(h, axis=1)[i]))
            p = np.vstack((p, t))
        print(u"预测准确度为：%f%%" % np.mean(np.float64(p == y_test.reshape(-1, 1)) * 100))
if __name__ == '__main__':
    model = OneVsAll()
    KLR = KLR()
    # data = model.loadmat_data(r"/home/donjy/datasets/data_digits.mat")
    # X, y = data['X'], data['y'].reshape(-1)
    data = pd.read_csv('/home/donjy/datasets/Indian_pines_corrected.csv', header=None).values
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    y_train = model.map_label(y_train)

    #计算核矩阵
    KernelMat = KLR.K(X_train)

    #保存计算得到的z
    all_z = model.calc_z(KLR,X_train,X_test,y_train,KernelMat)

    sigma = model.sigma
    np.savetxt("/home/donjy/datasets/gen_datas/Newton_Z_indian_{}.csv".format(sigma), all_z[:, :], delimiter=',')

    #读取保存的z
    # all_z = pd.read_csv(r"all_Zs_0.5.csv", delimiter=",", header=None).values
    model.calc_acc(all_z, X_test, y_test)