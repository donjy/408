import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,accuracy_score
class MLR(object):
    def __init__(self, batch_size=0, epochs=5000, learning_rate=0.5, reg_strength=1e-5, toler=1e-3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.toler = toler

    def fit(self, X, y):
        if (self.batch_size <= 0 or self.batch_size > X.shape[0]): self.batch_size = X.shape[0]
        self.y_unique = np.unique(y)
        self.W = np.zeros([X.shape[1], len(self.y_unique)])
        loss_epoch = []; loss = 0; epoch = 0
        while (epoch < self.epochs):
            loss_old = np.copy(loss)
            loss = self.sgd(X, y)
            diff = np.linalg.norm(loss - loss_old)
            epoch += 1
            if ((epoch) % 100 == 0 or epoch < 10 or epoch + 1 == self.epochs):
                print("Epoch: %s, Diff: %s" % (epoch , diff))
            loss_epoch.append(loss)
            if diff < self.toler:
                # print(loss_epoch)
                return
            # print(np.linalg.norm(self.W, axis=0))

    def predict(self, X):
        return self.y_unique[np.argmax(X.dot(self.W), axis=1)]

    def calculate_accuracy(self, y_test, y_pred):
        return np.mean(np.array(y_test == y_pred, dtype=float) * 100)

    def safe_log(self, x, minval=1e-8):
        return np.log(x.clip(min=minval))

    def calculate_loss(self, X, y):
        sample_size = X.shape[0]
        predictions = X.dot(self.W)  # Guess that in some case, it might be "predictions = X.dot(self.W) + b"
        predictions -= predictions.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        softmax = np.e ** predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        y_index = np.where(self.y_unique == np.array([y]).T)[1]
        loss = -self.safe_log(softmax[np.arange(len(softmax)), y_index]).sum() / sample_size
        # L2 正则
        loss += 0.5 * self.reg_strength * (self.W**2).sum()
        softmax[np.arange(len(softmax)), y_index] -= 1
        dW = X.T.dot(softmax) / sample_size + self.reg_strength * self.W
        return loss, dW

    def calculate_gradient(self, X, y, batch_size):
        import random
        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        return self.calculate_loss(X_batch, y_batch)

    def sgd(self, X, y):
        loss, dW = self.calculate_gradient(X, y, self.batch_size)
        self.W -= self.learning_rate * dW
        return loss


def loadmat_data(fileName):
    import scipy.io as spio
    return spio.loadmat(fileName)

def CrossValidation(X, y, model, splits=10, metric=mean_absolute_error):
    X, y = map(np.array, (X, y))
    quota, max_quota, min_quota = 0, 0, np.inf
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=splits, random_state=0, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        currentQuota = metric(y_test,y_predict)
        if(currentQuota > max_quota):
            max_quota = currentQuota
        elif(currentQuota < min_quota):
            min_quota = currentQuota
        quota += currentQuota/splits
        # print(metric.func_name + ':' + str(currentQuota))
    print("----------------------------------------")
    print("[mean accuracy] "  + ': ' + str(quota))
    print("[max accuracy] "  + ': ' + str(max_quota))
    print("[min accuracy] "  + ': ' + str(min_quota))
    return
if __name__ == '__main__':
    model = MLR()
    data = loadmat_data("E:\DataSets\data_digits.mat")
    X, y = data['X'], data['y'].reshape(-1)

    # data = pd.read_csv('E:\DataSets\ionosphere.csv', header=None).values
    # X = data[:, :-1]
    # y = data[:, -1]

    # # 交叉验证
    # X_train, y_train = X, y
    # CrossValidation(X_train, y_train, model, 5, accuracy_score)

    # 计算Accuracy
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = model.calculate_accuracy(y_test, y_pred)
    print("The Accuracy:{}".format(acc))
