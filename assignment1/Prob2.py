import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time

class Perceptron():
    '''This class is for train / test iris feature lib for classification work'''
    def __init__(self):
        self.filename_train = 'iris_train.csv'
        self.filename_test = 'iris_test.csv'
        self.data_train = pd.read_csv(self.filename_train)
        self.data_test = pd.read_csv(self.filename_test)
        self.x_train = np.vstack((self.data_train['Petallength'], self.data_train['Petalwidth'], np.ones(len(self.data_train['Petalwidth']))))
        self.x_test = np.vstack((self.data_test['Petallength'], self.data_test['Petalwidth'], np.ones(len(self.data_test['Petalwidth']))))
        self.y_train = self.data_train['label']
        self.y_test = self.data_test['label']
        self.w = np.zeros(3)

    def get_class(self):
        self.pos_len_train, self.pos_wid_train, self.neg_len_train, self.neg_wid_train = [], [], [], []
        self.pos_len_test, self.pos_wid_test, self.neg_len_test, self.neg_wid_test = [], [], [], []

        for i in range(len(self.y_train)):
            if self.y_train[i] == 1:
                self.pos_len_train.append(self.x_train[0][i])
                self.pos_wid_train.append(self.x_train[1][i])
            else:
                self.neg_len_train.append(self.x_train[0][i])
                self.neg_wid_train.append(self.x_train[1][i])

        for i in range(len(self.y_test)):
            if self.y_test[i] == 1:
                self.pos_len_test.append(self.x_test[0][i])
                self.pos_wid_test.append(self.x_test[1][i])
            else:
                self.neg_len_test.append(self.x_test[0][i])
                self.neg_wid_test.append(self.x_test[1][i])

    # def preception(self, x, y, w):
    #     # iteration = 0
    #     a = []
    #     for i in range(len(y)):
    #         if (np.dot(w, x[:,i]) * y[i]) <= 0:
    #             w += y[i] * x[:, i]
    # #             iteration += 1
    #             break
    #             ## should be some sentence to jump the for loop but not while
    #         else:
    #             a.append(i)
    #     return a, w
    def preception(self):
        # iteration = 0
        a = []
        for i in range(len(self.y_train)):
            if (np.dot(self.w, self.x_train[:,i]) * self.y_train[i]) <= 0:
                self.w += self.y_train[i] * self.x_train[:, i]
    #             iteration += 1
                break
                ## should be some sentence to jump the for loop but not while
            else:
                a.append(i)
        return a

    def train(self):
        self.iteration = 0
        while(1):
            self.iteration += 1
            print(self.iteration, self.w)
            a = self.preception()
            # a, self.w = self.preception(self.x_train, self.y_train, self.w)
            if len(a) == len(self.y_train):
                # print(a)
                break

    def test(self):
        result = np.zeros(len(self.y_test))
        for i in range(len(self.y_test)):
            if (np.dot(self.w, self.x_test[:,i]) * self.y_test[i] > 0):
                result[i] = 1
            else:
                result[i] = 0
        return result

def draw_line(w):
    x = np.linspace(1,7,100)
    y = np.zeros(100)
    for i in range(len(x)):
        y[i] = -(w[0]*x[i] + w[2]) / w[1]
    return x,y

def main():
    # filename_train, filename_test = 'mnist_train.csv', 'mnist_test.csv'
    Neuron = Perceptron()
    Neuron.get_class()

    a = time.time()

    # traim the model
    Neuron.train()

    b = time.time()

    print("The time to train the model is: ", b - a)

    # test the model
    result = Neuron.test()

    c = time.time()

    print("THe time to test the model is: ", c - b, "accuracy: ", sum(result)/len(result))

    # plot train data

    x, y = draw_line(Neuron.w)

    f, ax1 = plt.subplots()
    # ax1 = fig1.add_subplot(1, 1, 1, axisbg="1.0")
    ax1.scatter(Neuron.pos_len_train, Neuron.pos_wid_train, color = 'r', label='+1 labels')
    ax1.scatter(Neuron.neg_len_train, Neuron.neg_wid_train, color = 'g', label='-1 labels')
    ax1.plot(x, y, label = 'Preception classifier')
    ax1.set_title('Train set')
    ax1.set_xlabel('Petallength')
    ax1.set_ylabel('Petalwidth')
    ax1.legend()
    # x, y = draw_line(Neuron.w)
    # ax1.plot(x, y)

    f, ax2 = plt.subplots()
    # ax1 = fig1.add_subplot(1, 1, 1, axisbg="1.0")
    ax2.scatter(Neuron.pos_len_test, Neuron.pos_wid_test, color = 'r', label='+1 labels')
    ax2.scatter(Neuron.neg_len_test, Neuron.neg_wid_test, color = 'g', label='-1 labels')
    ax2.plot(x, y, label = 'Preception classifier')
    ax2.set_title('Test set')
    ax2.set_xlabel('Petallength')
    ax2.set_ylabel('Petalwidth')
    ax2.legend()
    # x, y = draw_line(Neuron.w)
    # ax2.plot(x, y)
    plt.show()

    # d = time.time()
    # print("THe time to get test data accracy is: ", d - c, " and the accracy is ", test_accuracy)


# THe time to get test data accracy is:  3373.176635980606  and the accracy is  0.6423773729562159


if __name__ == "__main__":
    main()
