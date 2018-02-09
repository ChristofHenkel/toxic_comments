import numpy
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from sklearn.metrics import log_loss
from train_model import ToxicComments
import tqdm
import pandas as pd
import os
from utilities import coverage
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import scipy

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_logits = ['logits_' + c for c in list_classes]

csvs_train = ['models/CNN/inception_2_2/train_logits_folded.csv',
              'models/RNN/pavel_baseline/train_logits_folded.csv',
              'models/CAPS/caps_first_test/train_logits/caps_first_testk0_e3.csv']

dfs = [pd.read_csv(csv) for csv in csvs_train]
xs = [df[list_logits].values for df in dfs]
n_models = len(csvs_train)

ys = [df[list_classes].values for df in dfs]

for i,_ in enumerate(csvs_train[1:]):
    assert np.array_equal(ys[0],ys[i])

X = np.hstack(xs)
Y = ys[0]

split_at = len(X)//10

kf = KFold(n_splits=10)
for train, valid in kf.split(X):
    X_train = X[train]
    Y_train = Y[train]
    X_valid = X[valid]
    Y_valid = Y[valid]

    #X_train = X[split_at:]
    #Y_train = Y[split_at:]
    #X_valid = X[:split_at]
    #Y_valid = Y[:split_at]

    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():

        x = tf.placeholder(shape=(None,n_models*len(list_classes)),dtype=tf.float32)
        y = tf.placeholder(shape=(None,6),dtype=tf.float32)

        h1 = layers.fully_connected(x, 64, activation_fn=tf.nn.elu)
        #h2 = layers.fully_connected(h1, 16, activation_fn=tf.nn.elu)
        logits = layers.fully_connected(h1,6,activation_fn=tf.nn.sigmoid)

        cost = tf.losses.log_loss(predictions=logits, labels=y)
        loss = binary_crossentropy(y, logits)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,momentum=0.9).minimize(loss)

    epochs = 50
    bsize = 1024

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):
            step = 0
            while step * bsize < len(X_train):
                batch_x = X_train[step * bsize:(step + 1) * bsize]
                batch_y = Y_train[step * bsize:(step + 1) * bsize]
                _ , logloss = sess.run([optimizer,cost], feed_dict={x:batch_x,y:batch_y})
                step += 1

            logloss_val, logits_val = sess.run([cost,logits], feed_dict={x: X_valid, y: Y_valid})
            print(logloss_val)




    def lloss(y_true,y_pred):
        l = 0
        for i in range(6):
            l += log_loss(y_true=y_true[:,i],y_pred=y_pred[:,i])
            l /= 6
        return l

    for x in xs:
        print(lloss(y_true=Y_valid,y_pred=x[valid]))

    print('------------->')
    print(lloss(y_true=Y_valid,y_pred=logits_val))
    print('using mean %s' %lloss(Y_valid,np.mean([x[valid] for x in xs],axis=0)))
    print('using geomean %s' % lloss(Y_valid, scipy.stats.gmean([x[valid] for x in xs], axis=0)))
    print('----------------------------')
    for x in xs:
        print(roc_auc_score(y_true=Y_valid,y_score=x[valid]))
    print('------------->')
    print(roc_auc_score(y_true=Y_valid,y_score=logits_val))
    print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[valid] for x in xs],axis=0)))
    print('using geomean %s' % roc_auc_score(Y_valid, scipy.stats.gmean([x[valid] for x in xs], axis=0)))
