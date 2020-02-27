import logging

import time
from numpy.linalg import norm
from sklearn import metrics

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
logger = logging.getLogger("smlr-ista")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG)

class SMLRIsta(object):
    def __init__(self, train_file=None, test_file=None,
                 t=1e-2, lamda=1., max_iter=100, alpha=0.5, tot=1e-3, use_sim_data=False):

        # ista hyper-parameter
        self.t = t
        self.lamda = lamda
        self.max_iter = max_iter
        self.alpha = alpha
        self.tot = tot

        # load dataset
        if use_sim_data:
            X, y = self.data_generate()
            self.X, self.test_x, self.y, self.test_y = train_test_split(
                X, y, test_size=0.33, random_state=1
            )
        else:
            if test_file is None:
                logger.info("###### using train test split")
                X, y = self.load_data(train_file)
                self.X, self.test_x, self.y, self.test_y = train_test_split(
                    X, y, test_size=0.33, random_state=1
                )
                del X, y
            else:
                self.X, self.y = self.load_data(train_file)
                self.test_x, self.test_y = self.load_data(test_file)
        logger.info("X: {}, y: {}".format(self.X.shape, self.y.shape))
        self.y = self.y.astype(int)
        self.test_y = self.test_y.astype(int)

        # smlr hyper-parameter
        self.k = len(set(self.y))

        # label shift(starting from 0)
        if self.y.min()==1:
            self.y -= 1; self.test_y -= 1

        # smlr parameter
        self.theta = np.zeros((self.X.shape[1], self.k))

    @staticmethod
    def data_generate():
        from settings import n_samples, n_features, n_classes
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features, n_informative=n_features, n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1, flip_y=0.1, random_state=3
        )
        # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, alpha=0.9, s=30, edgecolor='b')
        # plt.show()
        return X, y

    @staticmethod
    def load_data(file_name):
        logger.info("loading file " + file_name)
        data = np.loadtxt(file_name, dtype=np.float, delimiter=",", skiprows=1)
        X = data[:,:-1]
        y = data[:,-1]
        return X, y

    @staticmethod
    def prox(a, kappa):
        a_ = a.copy()
        a_[np.where((a_ <= kappa) & (a_ >= -kappa))] = 0
        a_[np.where(a_ > kappa)] -= kappa
        a_[np.where(a_ < - kappa)] += kappa
        return a_

    def minimize(self):
        """ proximal gradient descent """
        x = np.ones_like(self.theta)
        x_prox = np.zeros_like(self.theta)
        worse_cnt = 0
        early_stop_round = 3
        best_obj = np.inf
        best_theta = np.zeros_like(self.theta)
        for i in range(self.max_iter):
            g = self.grad(x)

            # back-tracking line search
            while True:
                x_nxt = x - self.t * g
                x_prox = self.prox(x_nxt, self.t * self.lamda)
                if self.line_search(x, x_prox, g, self.t): break
                else: self.t *= 0.5
            x = x_prox.copy()

            # early stop condition
            obj_cur = self.obj(x)
            if obj_cur < best_obj:
                worse_cnt = 0
                best_obj = obj_cur
                best_theta = x.copy()
            else:
                worse_cnt += 1
            if i > 0 and (norm(x - self.theta) < self.tot or worse_cnt >= early_stop_round):
                self.theta = best_theta
                break

            # if self.obj(x) > self.obj(self.theta): worse_cnt += 1
            # else: worse_cnt=0
            # if i > 0 and (norm(x-self.theta) < self.tot or worse_cnt >= 3): break

            self.theta = x.copy()
            logger.info("iter: {}".format(i))
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

    @staticmethod
    def safe_log(x, minval=1e-10):
        return np.log(x.clip(min=minval))

    @staticmethod
    def softmax(x):
        x -= x.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        p = np.e ** x
        p /= p.sum(axis=1).reshape([-1, 1])
        return p

    def obj(self, theta):
        """ objective of the differentiable function g """
        m = self.X.shape[0]
        margin = self.X.dot(theta)
        p = self.softmax(margin)
        loss_obj = -self.safe_log(p[np.arange(len(p)), self.y]).sum() / m
        return loss_obj

    def grad(self, theta):
        """ gradient of the differentiable function g """
        m = self.X.shape[0]
        margin = self.X.dot(theta)
        p = self.softmax(margin)
        p[np.arange(len(p)), self.y] -= 1
        dW = self.X.T.dot(p) / m
        return dW

    def predict(self):
        margin = self.test_x.dot(self.theta)
        p = self.softmax(margin)
        preds = p.argmax(axis=1).astype(float)
        logger.info("accuracy: {}".format(metrics.accuracy_score(self.test_y, preds)*100))
        return preds

if __name__ == '__main__':
    base = 'E:\\DataSets\\'
    # Note dataset header
    # train_path = base + 'Indian_pines_corrected.csv'; test_path = None
    train_path = base + 'EMP_Indina_profiles325.csv'; test_path = None

    smlr = SMLRIsta(
        train_file=train_path, test_file=test_path,
        t=1e-0, lamda=1e-3, max_iter=3000, alpha=0.5, use_sim_data=False
    )
    start = time.time()
    smlr.minimize()
    end = time.time()
    logger.info("runing time: {}".format(round(end -start, 2)))
    smlr.predict()

