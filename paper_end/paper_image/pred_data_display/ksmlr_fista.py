# -*- coding: utf-8 -*-
import logging
import pandas as pd
from numpy.linalg import norm
from sklearn import metrics
import spectral
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
logger = logging.getLogger("smlr-fista")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG)
class SMLRFista(object):
    def __init__(self, train_file=None, test_file=None,
                 t=1e-2, lamda=1., max_iter=100, alpha=0.2, tot=1e-3, sigma=2.0, img_name='out_imgs'):
        # fista hyper-parameter
        self.t = t
        self.lamda = lamda
        self.max_iter = max_iter
        self.alpha = alpha
        self.tot = tot
        self.sigma = sigma
        self.img_name = img_name
        # load dataset
        logger.info("loading file " + train_file)
        data = pd.read_csv(train_file, header=None)
        data = data.values
        data_D = data[:,:-1]
        data_L = data[:,-1]
        data_L = [int(lable - 1) for lable in data_L]
        # self.X, self.test_x, self.y, self.test_y = data_D, data_D, data_L, data_L
        self.X, self.test_x, self.y, self.test_y = train_test_split(data_D, data_L, test_size=0.2)

        # 为了方便成像，这里用所有数据进行测试。
        self.test_x = data_D
        self.test_y = data_L

        logger.info("The sigma:{}, alpha:{}".format(self.sigma, self.alpha))
        # print(self.y)
        # self.y = self.y.astype(int)
        # self.test_y = self.test_y.astype(int)

        # smlr hyper-parameter
        self.k = len(set(self.y))

        # label shift(starting from 0)
        if min(self.y)==1:
            self.y -= 1; self.test_y -= 1

        # smlr parameter
        self.y_unique = np.unique(self.y)
        self.alpha = np.zeros([self.X.shape[0], self.k])  # 初始化为1500X10
        self.Kernel_Mat = self.Kernel(self.X)

    @staticmethod
    def load_data(file_name):
        logger.info("loading file " + file_name)
        data = np.genfromtxt(file_name, dtype=np.float, delimiter=",")
        X = data[:,:-1]
        y = data[:,-1]
        return X, y
    # def load_data(file_name):
    #     import scipy.io as spio
    #     data = spio.loadmat(file_name)
    #     return data['X'], data['y'].reshape(-1)
    @staticmethod
    def prox(a, kappa):
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] -= kappa
        a_[np.where(a_ < - kappa)] += kappa
        return a_

    @staticmethod
    def safe_log(x, minval=1e-10):
        return np.log(x.clip(min=minval))

    @staticmethod
    def softmax(x):
        x -= x.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        p = np.e ** x
        p /= p.sum(axis=1).reshape([-1, 1])
        return p

    def Kernel(self, X):
        from mklaren.kernel.kinterface import Kinterface
        from mklaren.kernel.kernel import rbf_kernel
        K = Kinterface(data=X, kernel=rbf_kernel, kernel_args={"sigma": self.sigma})
        return K[:,:]

    def kernelMat(self,X1, X2):
        # sample size
        n1 = X1.shape[0]  # shape[0]查看行数
        n2 = X2.shape[0]
        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(-np.linalg.norm(X1[i]-X2[j])**2 / (2 * (self.sigma**2)))
        return K

    def minimize(self):
        # proximal gradient descent
        x = np.ones_like(self.alpha)
        x_prox = np.zeros_like(self.alpha)
        worse_cnt = 0
        early_stop_round = 3
        best_obj = np.inf
        best_alpha = np.zeros_like(self.alpha)
        for i in range(self.max_iter):
            y = x + (x - self.alpha) * (i - 1) / (i + 2)
            g = self.grad(y)

            # back-tracking line search
            while True:
                x_nxt = y - self.t * g
                x_prox = self.prox(x_nxt, self.t*self.lamda)
                if self.line_search(y, x_prox, g, self.t): break
                else: self.t *= 0.5

            # early stop condition
            obj_cur = self.obj(x)
            # logging.info("The Last loss:{}".format(obj_cur))
            if obj_cur < best_obj:
                worse_cnt = 0
                best_obj = obj_cur
                best_alpha = x.copy()
            else:
                worse_cnt += 1
            if i > 0 and (norm(x - self.alpha) < self.tot or worse_cnt >= early_stop_round):
                self.alpha = best_alpha
                break

            # if self.obj(x_prox) > self.obj(x): worse_cnt += 1
            # else: worse_cnt=0
            # if i > 0 and (norm(x_prox-x) < self.tot or worse_cnt >= 3): break

            self.alpha = x.copy()
            x = x_prox.copy()
            # logger.info("iter: {}".format(i))
            # logger.info("iter: {}, step: {}, obj: {}".format(i, self.t, self.obj(self.theta)))

    def line_search(self, x, z, g, t):
        """ return True if f_nxt <= f_cur """
        gamma = 1e-3
        delta_x = z - x
        f_nxt = self.obj(z)
        # f_cur = self.obj(x) + gamma*t*np.sum(delta_x*g)
        f_cur = self.obj(x) + np.sum(delta_x*g) + (1/(2*t))*np.sum(delta_x*delta_x)
        # logger.info("f_nxt: {}, f_cur: {}".format(f_nxt, f_cur))
        if f_nxt > f_cur:
            return False
        return True

    def obj(self, alpha):
        """ objective of the differentiable function g """
        m = self.X.shape[0]
        margin = self.Kernel_Mat.dot(alpha)   #(1500X10)
        p = self.softmax(margin)
        loss_obj = -self.safe_log(p[np.arange(len(p)), self.y]).sum() / m
        # logging.info("The loss:{}".format(loss_obj))
        return loss_obj

    def grad(self, alpha):
        """ gradient of the differentiable function g """
        m = self.X.shape[0]
        margin = self.Kernel_Mat.dot(alpha)
        p = self.softmax(margin)
        p[np.arange(len(p)), self.y] -= 1
        d_alpha = self.Kernel_Mat.T.dot(p) / m
        return d_alpha

    def predict(self):
        kernel = self.kernelMat(self.test_x, self.X)
        margin = kernel.dot(self.alpha)
        p = self.softmax(margin)
        preds = p.argmax(axis=1).astype(int)
        logger.info("accuracy: {}".format(metrics.accuracy_score(self.test_y, preds) * 100))

        # 打印图像，对比。
        from scipy.io import loadmat
        # output_image = loadmat('/home/cs/donjy/datasets/matlabel/PaviaU_gt.mat')['paviaU_gt']
        # output_image = loadmat('/root/donjy/datasets/matlabel/Indian_pines_gt.mat')['indian_pines_gt']
        output_image = loadmat('E:\\DataSets\\matlabel\\Indian_pines_gt.mat')['indian_pines_gt']
        # 打印图像的时候label 在预测时候减了1，这里要加回去。
        predict_label = [int(lable + 1) for lable in preds]
        # 将预测的结果匹配到图像中

        new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
        k = 0
        for i in range(output_image.shape[0]):
            for j in range(output_image.shape[1]):
                if output_image[i][j] != 0 :
                    new_show[i][j] = predict_label[k]
                    k +=1
        #颜色
        indian_color =np.array(
            [[255,255,255],
             [184,40,99],
             [74,77,145],
             [35,102,193],
             [238,110,105],
             [117,249,76],
             [114,251,253],
             [126,196,59],
             [234,65,247],
             [141,79,77],
             [183,40,99],
             [0,39,245],[90, 196, 111],
             [33, 21, 32],
             [132, 121, 132],
             [63, 91, 12]
             ])

        # 展示地物
        ground_truth = spectral.imshow(classes = output_image.astype(int),colors=indian_color, figsize =(9,9))
        #
        ground_predict = spectral.imshow(classes = new_show.astype(int), colors=indian_color,figsize =(9,9))

        plt.axis('off')  # 去掉坐标
        plt.savefig("E:\\DataSets\\matlabel\\imgs/{}.png".format(self.img_name))
        # plt.show(ground_truth)
        # plt.show(ground_predict)
        spectral.save_rgb("/home/cs/donjy/{}.png".format(self.img_name),new_show.astype(int),colors=spectral.spy_colors,format='png')

if __name__ == '__main__':
    """
        1. the parameter initial value has a great effect.
        2. the hyper-parameter lamda also has a great effect.
    """

    # base = 'E:\\DataSets\\01\\'
    # base = '/root/donjy/datasets/matlabel/'
    # base = '/home/cs/donjy/datasets/matlabel/'
    base = 'E:\\DataSets\\matlabel\\'
    # Note dataset header


    ####### traing==test dataset.....
    # train_path = base + 'Indian_pines_corrected.csv'
    train_path = base + 'Indian_pines_corrected.csv'
    test_path = None
    # train_path = base + 'YLB_48_42_train_sk.csv'; test_path = base + 'YLB_48_42_test_sk.csv'
    sigma_list =[0.5, 0.3, 1.5, 0.2, 0.05]

    for i in sigma_list:
        smlr = SMLRFista(
            train_file=train_path, test_file=test_path,
            t=1e-0, lamda=1e-5, max_iter=3000, alpha=0.5, sigma=i , img_name= 'ground_predict_sigma:{}'.format(i)
        )

        smlr.minimize()
        smlr.predict()

