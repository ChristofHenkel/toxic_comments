import numpy
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from sklearn.metrics import log_loss
import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import scipy
from utilities import corr_matrix

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_logits = ['logits_' + c for c in list_classes]

csvs_train = ['models/CNN/inception5_slim/train_logits_folded.csv',
              'models/RNN/pavel_baseline/train_logits_folded.csv',
              'models/CAPS/caps_first_test/train_logits/caps_first_testk0_e3.csv']

dfs = [pd.read_csv(csv) for csv in csvs_train]
xs = [df[list_logits].values for df in dfs]
n_models = len(csvs_train)

print('Corr matrix')
print(corr_matrix(xs))
print(' ')


df = dfs[1].copy()
for logit in list_logits:
    df[logit] = df[logit].map(lambda x: 0 if x < 0.02 else 1)

print(roc_auc_score(y_true=df[list_classes].values,y_score=df[list_logits].values))

"""
graph = tf.Graph()
with graph.as_default():

    X = tf.Variable(df[list_logits].values,trainable=False, dtype=tf.float32)
    Y = tf.Variable(df[list_classes].values,trainable=False,dtype=tf.float32)
    alpha = tf.Variable(0.4,dtype=tf.float32)
    res = tf.divide(tf.add(tf.sign(tf.subtract(X,alpha)),1),2)
    loss = tf.losses.log_loss(Y,res)

    optimizer = tf.train.AdamOptimizer.minimize(loss)

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10):
        l, _, a= sess.run([loss,optimizer,alpha])
        print(l)
        print(a)
"""