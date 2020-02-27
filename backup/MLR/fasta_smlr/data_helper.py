import logging
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger("data-helper")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class DataHelper(object):

    def __init__(self, train_file=None, test_file=None):
        # load dataset
        if test_file is None:
            logger.info("###### using train test split")
            X, y = self._load_data(train_file)
            self.X, self.x_test, self.y, self.y_test = train_test_split(
                X, y, test_size=0.33, random_state=1
            )
            del X, y
        else:
            self.X, self.y = self._load_data(train_file)
            self.x_test, self.y_test = self._load_data(test_file)

        logger.info("X: {}, y: {}".format(self.X.shape, self.y.shape))
        self.y = self.y.astype(int)
        self.y_test = self.y_test.astype(int)

        # smlr hyper-parameter
        self.k = len(set(self.y))

        # label shift(starting from 0)
        if self.y.min()==1:
            self.y -= 1; self.y_test -= 1


    @staticmethod
    def _load_data(file_name):
        logger.info("loading file " + file_name)
        data = np.loadtxt(file_name, dtype=np.float, delimiter=",", skiprows=1)
        X = data[:,:-1]
        y = data[:,-1]
        return X, y