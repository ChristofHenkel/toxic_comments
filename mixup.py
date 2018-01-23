"""
Exploration of mixup
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730
https://arxiv.org/abs/1710.09412

"""

import numpy as np


def mixup_old(alpha, batch_size, batch_x, batch_y):
    lam = np.random.beta(alpha, alpha, batch_size)
    x_weight = lam.reshape(batch_size, 1)
    y_weight = lam.reshape(batch_size, 1)
    index = np.random.permutation(batch_size)
    x1, x2 = batch_x, batch_x[index]
    x = x1 * x_weight + x2 * (1 - x_weight)
    y1, y2 = batch_y, batch_y[index]
    y = y1 * y_weight + y2 * (1 - y_weight)
    return x, y


def mixup(alpha, x1, x2, y1, y2):
    lam = np.random.beta(alpha, alpha)
    x = x1 * lam + x2 * (1 - lam)
    y = y1 * lam + y2 * (1 - lam)
    return x, y


def augment_with_mixup(X, Y, alpha, portion):
    iters = len(X) // portion
    X_new = np.zeros((iters,X.shape[1]))
    Y_new = np.zeros((iters,Y.shape[1]))
    for i in iters:
        indices = [k for k, x in enumerate(X)]
        ind1, ind2 = np.random.choice(indices,2,replace=False)
        X_new[i], Y_new[i] = mixup(alpha, X[ind1], X[ind2],Y[ind1], Y[ind2])
    res = np.random.shuffle((np.concatenate((X,X_new)),np.concatenate((Y,Y_new))))
    return res[0], res[1]
