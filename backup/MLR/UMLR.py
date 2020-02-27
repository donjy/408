#!/usr/bin/python
# -*- coding: utf-8 -*-
# Multinomial logistic regression experiment
from __future__ import division, print_function
from operator import itemgetter
from sklearn.metrics import mean_absolute_error,accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import scipy.io as spio

def ReadFile(TRAINFILE,TESTFILE=None):
    with open(TRAINFILE,'rb') as dataSetSource:
        dataSet = pd.read_csv(dataSetSource, header=None)
        if(TESTFILE is not None):
            testDataSetSource = open(TESTFILE, 'rb')
            testDataSet = pd.read_csv(testDataSetSource, header=None)
            testDataSetSource.close()
            return dataSet.iloc[:,:-1], dataSet.iloc[:,-1], testDataSet.iloc[:, :-1], testDataSet.iloc[:, -1]
    return dataSet.iloc[:, :-1], dataSet.iloc[:, -1]


def plot(y_true, y_predict):
    y_true, y_predict = list(y_true), list(y_predict)
    x = [i for i in range(len(y_true))]
    y_true.sort()
    y_predict.sort()
    plt.plot(x, y_true)
    plt.scatter(x, y_predict)
    plt.show()
    return


def Remind():

    root = Tkinter.Tk()
    root.attributes("-topmost",1)
    root.withdraw()
    tkMessageBox.showinfo("Notify", "End of Run!")
    return


def PickSample(X, y, pick_size, softmax, metric):
    uniqueClass = np.unique(y)
    classSampleNum = [[] for i in range(len(uniqueClass))]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    #     random.seed(1)

    for i in range(len(y)):
        classSampleNum[uniqueClass.tolist().index(y[i])].append(i)

    for i in range(len(classSampleNum)):
        random_indices = random.sample(classSampleNum[i], np.clip(pick_size, 0 + 1, len(classSampleNum[i]) - 1))
        test_indices = list(set(classSampleNum[i]).difference(set(random_indices)))
        X_train.extend(X[random_indices])
        y_train.extend(y[random_indices])
        X_test.extend(X[test_indices])
        y_test.extend(y[test_indices])

    softmax.train(np.array(X_train), np.array(y_train))
    predictValue = softmax.predict(np.array(X_test))
    print("Accuracy" + str(metric(predictValue, np.array(y_test))))
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def PickClass(X, y, pick_size=2, random_state=None):
    X, y = map(np.array, (X, y))
    X_picked, y_picked = [], []
    y_unique = np.unique(y)
    if(random_state is not None):
        random.seed(random_state)
    y_picked_class = random.sample(y_unique, pick_size)
    for i in range(len(y)):
        if(y[i] in y_picked_class):
            X_picked.append(X[i])
            y_picked.append(y[i])
    return X_picked, y_picked


def TrainTestValidation(X_train, X_test, y_train, y_test, model, metric=mean_absolute_error):
    X_train, X_test, y_train, y_test = map(np.array, (X_train, X_test, y_train, y_test))
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    quota = metric(y_predict, y_test)
    print(metric.func_name + ': ' + str(quota))
    Remind(), plot(y_test, y_predict)
    return


def SplitValidation(X, y, model, test_size=0.2, metric=mean_absolute_error):
    X, y = map(np.array, (X, y))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    quota = metric(y_predict, y_test)
    print(metric.func_name + ': ' + str(quota))
    Remind(), plot(y_test, y_predict)
    return


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
        print(metric.func_name + ':' + str(currentQuota))
    print("----------------------------------------")
    print("[mean] " + metric.func_name + ': ' + str(quota))
    print("[max] " + metric.func_name + ': ' + str(max_quota))
    print("[min] " + metric.func_name + ': ' + str(min_quota))
    return


class MultinomialLogisticRegression:
    def __init__(self, batch_size=0, epochs=1000, learning_rate=1e-2, reg_strength=0*1e-5, eta=0*1e-5, weight_update='adam'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.eta = eta
        self.weight_update = weight_update
        return

    def fit(self, X, y):
        if(self.batch_size <= 0 or self.batch_size > X.shape[0]): self.batch_size = X.shape[0]
        self.y_unique = np.unique(y)
        # self.W = np.ones([X.shape[1], len(self.y_unique)])
        self.W = np.zeros([X.shape[1], len(self.y_unique)])
        # Another random way is "self.W = np.random.randn(n_features, n_classes) / np.sqrt(n_features/2)"
        config = {'reg_strength': self.reg_strength, 'batch_size': self.batch_size, 'learning_rate': self.learning_rate, 'eps': 1e-8, 'decay_rate': 0.99, 'momentum': 0.9, 'cache': None, 'beta_1': 0.9, 'beta_2':0.999, 'velocity': np.zeros(self.W.shape)}
        loss_epoch = []
        for epoch in range(self.epochs):
            loss, config = getattr(globals()['MultinomialLogisticRegression'], self.weight_update)(self, X, y, config)
            if((epoch + 1) % 100 == 0 or epoch < 10 or epoch + 1 == self.epochs): print("Epoch: %s, Loss: %s" % (epoch + 1, loss))

            #print(np.linalg.norm(self.W, axis=0))
            loss_epoch.append(loss)
        # norm_ = np.linalg.norm(self.W, axis=0)
        # pd.DataFrame(norm_).to_csv("D:/Desktop/vy_.csv", index=False, header=None)
        # pylib.lists_to_file(r"D:/Desktop/new.csv", loss_epoch)
        # sys.exit(-1)
        return

    def predict(self, X):
        return self.y_unique[np.argmax(X.dot(self.W), 1)]

    def calculate_loss(self, X, y):
        sample_size = X.shape[0]
        predictions = X.dot(self.W)  # Guess that in some case, it might be "predictions = X.dot(self.W) + b"
        predictions -= predictions.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
        softmax = np.e**predictions
        softmax /= softmax.sum(axis=1).reshape([-1, 1])
        y_index = np.where(self.y_unique == np.array([y]).T)[1]
        loss = -self.safe_log(softmax[np.arange(len(softmax)), y_index]).sum() / sample_size
        loss += 0.5 * self.reg_strength * (self.W**2).sum()
        softmax[np.arange(len(softmax)), y_index] -= 1
        dW = X.T.dot(softmax) / sample_size + self.reg_strength * self.W
        return loss, dW

    def safe_log(self, x, minval=0.0000000001):
        return np.log(x.clip(min=minval))

    def adam(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'eps', 'beta_1', 'beta_2')(config)
        learning_rate, batch_size, reg_strength, eps, beta_1, beta_2 = items
        config.setdefault('t', 0)
        config.setdefault('m', np.zeros(self.W.shape))
        config.setdefault('v', np.zeros(self.W.shape))

        loss, dW = self.consecutive_sample(X, y, batch_size)

        config['t'] += 1
        config['m'] = config['m']*beta_1 + (1-beta_1)*dW
        config['v'] = config['v']*beta_2 + (1-beta_2)*dW**2
        m = config['m']/(1-beta_1**config['t'])
        v = config['v']/(1-beta_2**config['t'])
        self.W -= learning_rate*m/(np.sqrt(v)+eps)
        return loss, config

    def sgd(self, X, y, config):
        learning_rate, batch_size, reg_strength = itemgetter('learning_rate', 'batch_size', 'reg_strength')(config)

        loss, dW = self.calculate_gradient(X, y, self.batch_size)

        self.W -= learning_rate * dW
        return loss, config

    def sgd_with_momentum(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'momentum')(config)
        learning_rate, batch_size, reg_strength, momentum = items

        loss, dW = self.calculate_gradient(X, y, self.batch_size)

        config['velocity'] = momentum*config['velocity'] - learning_rate*dW
        self.W += config['velocity']
        return loss, config

    def rms_prop(self, X, y, config):
        items = itemgetter('learning_rate', 'batch_size', 'reg_strength', 'decay_rate', 'eps', 'cache')(config)
        learning_rate, batch_size, reg_strength, decay_rate, eps, cache = items

        loss, dW = self.calculate_gradient(X, y, self.batch_size)

        cache = np.zeros(dW.shape) if cache == None else cache
        cache = decay_rate * cache + (1-decay_rate) * dW**2
        config['cache'] = cache

        self.W -= learning_rate * dW / (np.sqrt(cache) + eps)
        return loss, config

    def calculate_gradient(self, X, y, batch_size):
        random_indices = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[random_indices]
        y_batch = y[random_indices]
        return self.calculate_loss(X_batch, y_batch)

    def consecutive_sample(self, X, y, batch_size):
        import random
        try:
            if self.start_index + batch_size < len(X):
                self.start_index += batch_size
            else:
                self.start_index = 0
        except:
            self.start_index = 0
            self.random_indices = range(len(X))
            random.shuffle(self.random_indices)
        return self.calculate_loss(X[self.random_indices[self.start_index:self.start_index+batch_size]], y[self.random_indices[self.start_index:self.start_index+batch_size]])

    def SaveModel(self, OUTPUTFILE):
        import pickle
        with open(OUTPUTFILE, 'wb') as fileSource:
            pickle.dump(self.W, fileSource)
        return


def main():
    data = spio.loadmat("E:\DataSets\data_digits.mat")
    X, y = data['X'], data['y'].reshape(-1)
    X_train, y_train = X,y
    # X_train, y_train = PickClass(X_train, y_train, pick_size=2, random_state=5)
    sfm = MultinomialLogisticRegression(batch_size=0, epochs=1000, learning_rate=1*1e-2, reg_strength=0*1e-5, eta=10*1e-5, weight_update='adam')
    CrossValidation(X_train, y_train, sfm, 2, accuracy_score)
    # SplitValidation(X_train, y_train, sfm, test_size=0.2, metric=accuracy_score)
    # TrainTestValidation(X_train, X_test, y_train, y_test, sfm, metric=accuracy_score)
    return exit(0)


if (__name__ == "__main__"):
    main()
