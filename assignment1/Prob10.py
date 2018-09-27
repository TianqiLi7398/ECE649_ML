import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import multivariate_normal
import time
# import matplotlib.pyplot as plt


class BayesN():
    '''this class is for Bayesian Network's classification'''
    # filename_train, filename_test = 'mnist_train.csv', 'mnist_test.csv'

    def __init__(self):
        filename_train, filename_test = 'mnist_train.csv', 'mnist_test.csv'

        data_train = pd.read_csv(filename_train)
        data_test = pd.read_csv(filename_test)
        self.y_train = data_train.values[:, 0]
        self.x_train = data_train.values[:, 1:]
        self.y_test = data_test.values[:, 0]
        self.x_test = data_test.values[:, 1:]
        self.smooth = 0.0001  # for singularity of sigma
        self.means = []
        self.covs = []
        self.py = np.zeros(10)
        self.px_y = np.zeros((10, 784))
#         self.Py()

#         self.get_mean()

    def get_mean(self):
        for i in range(10):
            # X = self.x_train[y_train == i]
            # for j in range(784):

            self.means.append(np.mean(self.x_train[self.y_train == i], axis=0))
            self.covs.append(np.cov(self.x_train[self.y_train == i].T))

    def Py(self):
        count = np.unique(self.y_train, return_counts=1)
        for i in range(10):
            self.py[i] = count[1][i] / sum(count[1])

    def Px_y(self):
        for i in range(10):
            self.means.append(
                np.mean(self.x_train[self.y_train == i], axis=0) / 256)
#             self.means.append(np.mean(self.x_train[self.y_train == i], axis=0)/256)

        # for i in range(10):
        #     a = self.x_train[self.y_train == i]
        #     for j in range(784):
        #         self.px_y[i, j] = (
        #             784 - np.count_nonzero(a[:, j] == 0)) / len(a)


# complicated case
    # def likelihood_log(self, x, y):
        # prob = 0
        # for i in range(784):
        #     # prob =+ multivariate_normal.logpdf(x[i], mean = self.means[y][i], cov = self.covs[y][i,i] + self.smooth)
        #     prob += -0.5 * np.log(self.covs[y][i, i] + self.smooth) - 0.5 * (
        #         x[i] - self.means[y][i]) ** 2 / (self.covs[y][i, i] + self.smooth)
        # return prob

    def likelihood_log(self, x, y):
        prob = 0
        for i in range(784):
            if x[i] == 0:
                prob += np.log(1 - self.means[y][i] - self.smooth)
            else:
                prob += np.log(self.means[y][i] + self.smooth)
            # if x[i] == 0:
            #     prob += np.log(1 -self.px_y[y, i])
            # else:
            #     prob += np.log(self.px_y[y, i])
        return prob

    def predict(self, xset):
        # here xset shoud be a vector 784*1
        probs = np.zeros(10)
        for i in range(10):
            probs[i] = np.log(self.py[i]) + self.likelihood_log(xset, i)
            # probs[i] += self.likelihood_log(xset, i)
#         print(probs)
#         return np.argmax(probs)
        return np.argmax(probs)

    def get_accuracy(self):
        # data number is y.shape[0]
        #         x, y = self.x_test[:100], self.y_test[:100]
        x, y = self.x_test, self.y_test
        results = []
        for i in range(y.shape[0]):
            #             a = self.predict(x[i])
            #             a = y[i] == self.predict(x[i])
            b = time.time()
            a = self.predict(x[i])
            print(i, 'the predict is ', y[
                  i], a, 'time ', time.time() - b, i / len(x), "%")
            results.append(a == y[i])
        return results.count(True) / y.shape[0]


def main():
    # filename_train, filename_test = 'mnist_train.csv', 'mnist_test.csv'
    BN = BayesN()

    a = time.time()

    # get p(y)
    BN.Py()

    b = time.time()

    print("The time to get P(y) is: ", b - a)
    # get the mean & cov of the model
    # BN.get_mean()
    BN.Px_y()

    c = time.time()

    print("THe time to get P(x|y) is: ", c - b)

    # test the model
    test_accuracy = BN.get_accuracy()
    # test_accuracy = BN.test()

    d = time.time()
    print("THe time to get test data accracy is: ",
          d - c, " and the accracy is ", test_accuracy)


# THe time to get test data accracy is:  3373.176635980606  and the
# accracy is  0.6423773729562159


if __name__ == "__main__":
    main()
