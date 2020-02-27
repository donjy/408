import logging

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from fasta_smlr import fasta_smlr, softmax
import numpy as np
import pandas as pd

logger = logging.getLogger("data-helper")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class DataHelper(object):

    def __init__(self, train_file=None, test_file=None, use_sim_data=False):
        # load dataset

        if use_sim_data:
            X, y = self.data_generate()
            self.X, self.x_test, self.y, self.y_test = train_test_split(
                X, y, test_size=0.33, random_state=1
            )
        else:
            if test_file is None:
                logger.info("###### using train test split")
                X, y = self._load_data(train_file)
                self.X, self.x_test, self.y, self.y_test = train_test_split(
                    X, y, test_size=0.33, random_state=1
                )
                del X, y
            else:
                logger.info("##### using train file and test file")
                self.X, self.y = self._load_data(train_file)
                self.x_test, self.y_test = self._load_data(test_file)

        self.y = self.y.astype(int)
        self.y_test = self.y_test.astype(int)

        # smlr hyper-parameter
        self.k = len(set(self.y))

        # label shift(starting from 0)
        if self.y.min()==1:
            self.y -= 1; self.y_test -= 1

        self.train = np.hstack((self.X, self.y.reshape(-1, 1)))
        self.test = np.hstack((self.x_test, self.y_test.reshape(-1, 1)))

    @staticmethod
    def data_generate():
        n_samples, n_features, n_classes = 16384, 500, 3
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
    def _load_data(file_name):
        logger.info("loading file " + file_name)
        data = pd.read_csv(file_name)
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        return X, y

base = 'F:\\MyDocuments\\2016-09-Hyperspectral\\Classification\\'
train_path = base + 'segment-challenge.csv'; test_path = base + 'segment-test.csv'
# train_path = base + 'waveform-+noise-train.csv'
# test_path = base + 'waveform-+noise-test.csv'
# train_path = base + 'COIL20.csv'; test_path = None
# train_path = base + 'MNIST.csv'; test_path = None
# train_path = base + 'GTcropped32_32_zscore_sk.csv'; test_path = None
# train_path = 'F:\\iris.csv'; test_path = None
# train_path = 'F:\\glass_op.csv'; test_path = None
# train_path = 'F:\\lung.csv'; test_path = None # 1e-1
# train_path = 'F:\\MNIST_zscore.csv'; test_path = None
# train_path = base + 'mnist_train_zscore_sk.csv'; test_path = base + 'mnist_test_zscore_sk.csv'
# train_path = base + 'YLB_96_84_train_zscore_sk.csv'; test_path = base + 'YLB_96_84_test_zscore_sk.csv'
# train_path = base + 'YLB_48_42_train_sk.csv'; test_path = base + 'YLB_48_42_test_sk.csv'

helper = DataHelper(train_path, test_path, use_sim_data=False)
train, test = helper.train, helper.test
logger.info("train: {}, test: {}".format(train.shape, test.shape))

A, b = train[:, :-1], train[:, -1].astype(int)

K = helper.k
lmd = 1e-5            # 2e-3, 1e-4 # 0.944792973651
M, N = A.shape

# The initial iterate:  a guess at the solution
x0 = np.ones((N,K)) # smlr parameter

# OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = {}
# opts.tol = 1e-8;  # Use super strict tolerance
opts['recordObjective'] = True  # Record the objective function so we can plot it
opts['verbose'] = False
opts['stringHeader'] = '\t'     # Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer.
opts['tau'] = 10 # 0.8
opts['maxIters'] = 1000

# Call the solver 3 times
sol, outs = fasta_smlr(A, A.T, b, lmd, x0, opts)

### evaluation
X, y = test[:, :-1], test[:, -1].astype(int)

margin = X.dot(sol)
p = softmax(margin)
preds = p.argmax(axis=1).astype(float)
print("accuracy: {}".format(metrics.accuracy_score(y, preds)*100))
print("running time: {}".format(round(outs['runningTime'], 2)))
