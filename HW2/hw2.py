'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg


# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    loss = 0.0
    for i in range(n):
        s = x[:, i]
        s = np.reshape(s, (feat_dim, 1))
        scores = W.dot(s) + b
        total = 0
        target = 0
        for j in range(num_class):
            total += np.exp(scores[j][0])
            if j == y[i]:
                target = np.exp(scores[y[i]][0])
        target /= total
        loss += -np.log(target)
    return loss
    pass


########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    loss = 0.0
    for i in range(n):
        s = x[:, i]
        s = np.reshape(s, (feat_dim, 1))
        scores = W.dot(s) + b
        correct_class_score = scores[y[i]][0]
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j][0] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
    loss /= n
    return loss
    pass


########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, k):
    n_train_sample = X_train.shape[0]
    n_test_sample = X_test.shape[0]
    testXtest = np.sum(np.square(X_train), axis=1).reshape(-1, 1)
    trainXtrain = np.sum(np.square(X_test), axis = 1).reshape(-1, 1)
    testXtrain = np.matmul(np.reshape(X_train, (n_train_sample, -1)), np.reshape(X_test, (n_test_sample, -1)).T)
    dists = testXtest + trainXtrain.T - 2*testXtrain

    y_pred, _ = stats.mode(y_train[np.argsort(dists, axis=0)[range(k), :]])
    accuracy = np.sum(np.reshape(y_pred, (-1,)) == y_test.reshape(-1, )) / n_test_sample
    return accuracy
    pass


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    # W has shape (num_class, feat_dim), b has shape (num_class,)

    # score
    s = x @ W.T + b
    # score has shape (n_test, num_class)

    # get argmax over class dim
    res = np.argmax(s, axis=1)

    # get accuracy
    accuracy = (res == y).astype('uint8').sum() / n_test

    return accuracy

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1

print('number of classes: ', num_class, ' sigma for data scatter:', sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'softmax'

if classifiers == 'knn':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class) * 100, '%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class) * 100, '%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, k) * 100
          , '% for k value of ', k)
