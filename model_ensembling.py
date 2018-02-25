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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.model_selection import KFold
import xgboost as xgb
from xgboost import XGBRegressor
import scipy
from utilities import corr_matrix, logloss
from global_variables import LIST_LOGITS, LIST_CLASSES

classifiers = ['xgb','lr','nn'] #nn, catboost
do_prediction = False
fn_out = 'models/ENSAMBLES/e3/'
# Todo add switch for per class or all logits

#csvs_train = ['models/CNN/inception2_slim/inception2_slim_baggin_logits_folded.csv',
#              'models/NBSVM/slim/nbsvm_prediction_valid.csv',
#              #'models/RNN/pavel_attention_slim2/baggin_logits_folded.csv',
#              'models/RNN/pavel_all_outs_slim/birnn_all_outs_slim_baggin_logits_folded.csv'
#              ]

csvs_train = ['models/CNN/inception2_slim/l2_train_data.csv',
              'models/NBSVM/slim/l2_train_data.csv',
              #'models/RNN/pavel_attention_slim2/baggin_logits_folded.csv',
              'models/RNN/pavel_all_outs_slim/l2_train_data.csv',
              'models/CAPS/cudrnn_caps_slim/l2_train_data.csv']
csvs_valid = ['models/CNN/inception2_slim/inception2_slim_baggin_logits_folded.csv',
              'models/NBSVM/slim/nbsvm_prediction_valid.csv',
              #'models/RNN/pavel_attention_slim2/baggin_logits_folded.csv',
              'models/RNN/pavel_all_outs_slim/birnn_all_outs_slim_baggin_logits_folded.csv',
              'models/CAPS/cudrnn_caps_slim/l2_valid_data.csv']

csvs_test = ['models/CNN/inception2_slim/inception2_slim.csv',
             'models/NBSVM/slim/nbsvm_submission.csv',
             #'models/RNN/pavel_attention_slim2/test_data_folded.csv',
             'models/RNN/pavel_all_outs_slim/birnn_all_outs.csv',
             'models/CAPS/cudrnn_caps_slim/cudrnn_caps_slim_test_data.csv']


def get_values(csv_files, columns, hstack = False, with_labels=True):
    dfs = [pd.read_csv(csv) for csv in csv_files]
    xs = [df[columns].values for df in dfs]
    if hstack:
        X = np.hstack(xs)
    else:
        X = np.concatenate([xs])
        X = X.transpose([1, 0, 2])

    if with_labels:
        ys = [df[LIST_CLASSES].values for df in dfs]

        for i, _ in enumerate(csv_files[1:]):
            assert np.array_equal(ys[0], ys[i])

        Y = ys[0]
        return X, Y
    else:
        return X

X_train, Y_train = get_values(csvs_train,columns=LIST_LOGITS,hstack=False,with_labels=True)
X_valid, Y_valid = get_values(csvs_valid,columns=LIST_LOGITS,hstack=False,with_labels=True)
X_test = get_values(csvs_train,columns=LIST_CLASSES,hstack=False,with_labels=False)


print('Corr matrix')
print(corr_matrix(list(X_train.transpose([1, 0, 2]))))
print(' ')


if 'xgb' in classifiers:
    xgb_clfs = []
    for i in range(6):
        print('fitting xgb on %s' %LIST_CLASSES[i])
        clf = XGBRegressor(objective='reg:logistic', max_depth=2, n_estimators=100, learning_rate=0.1, subsample=0.8,
                           min_child_weight=3)
        clf.fit(X_train[:, :, i], Y_train[:, i])
        xgb_clfs.append(clf)


if 'lr' in classifiers:
    lr_clfs = []
    for i in range(6):
        print('fitting logistic regression on %s' % LIST_CLASSES[i])
        clf_lr = LogisticRegression()
        clf_lr.fit(X_train[:, :, i], Y_train[:, i])
        lr_clfs.append(clf_lr)

if 'nn' in classifiers:
    clf_nn = MLPClassifier(solver='adam', batch_size=1024, hidden_layer_sizes=(64,), max_iter=150, verbose=True,
                           tol=0.000001)
    clf_nn.loss = 'log_loss'
    X_train2 = X_train.reshape((-1, len(csvs_train) * len(LIST_CLASSES)))
    clf_nn.fit(X_train2, Y_train)



preds_cat = np.zeros((len(X_valid), 6))
preds_xgb = np.zeros((len(X_valid), 6))
preds_logistic = np.zeros((len(X_valid), 6))

for i in range(6):
    preds_xgb[:, i] = xgb_clfs[i].predict(X_valid[:, :, i])
    preds_logistic[:, i] = lr_clfs[i].predict_proba(X_valid[:, :, i])[:, 1]
preds_nn = clf_nn.predict_proba(X_valid.reshape((-1, len(csvs_train) * len(LIST_CLASSES))))


if do_prediction:

    res = np.zeros((len(X_test),6))
    preds_xgb_test = np.zeros((len(X_test), 6))
    preds_logistic_test = np.zeros((len(X_test), 6))


    #preds_cat[:, i] = fit_model.predict_proba(X_valid[:,:,i])[:, 1]

    if do_prediction:
        preds_xgb_test[:, i] = clf.predict(X_test[:, :, i])
        preds_logistic_test[:, i] = clf2.predict_proba(X_test[:, :, i])[:, 1]
        #res[:, i] = fit_model.predict_proba(X_test)[:, 1]
        #print('using catboost %s' % lloss(Y_valid, preds_cat))

    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    #sample_submission[LIST_CLASSES] = res
    #sample_submission[LIST_CLASSES] = preds_xgb
    sample_submission[LIST_CLASSES] = preds_logistic_test
    if not os.path.exists(fn_out):
        os.mkdir(fn_out)

    #fn = fn_out + 'cat_boost_submission.csv'
    #fn = fn_out + 'xgb_blend_submission.csv'
    fn = fn_out + 'stacked_logistic_submission.csv'
    sample_submission.to_csv(fn, index=False)

    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    #sample_submission[LIST_CLASSES] = res
    #sample_submission[LIST_CLASSES] = preds_xgb
    sample_submission[LIST_CLASSES] = preds_xgb_test
    if not os.path.exists(fn_out):
        os.mkdir(fn_out)

    #fn = fn_out + 'cat_boost_submission.csv'
    #fn = fn_out + 'xgb_blend_submission.csv'
    fn = fn_out + 'stacked_xgb_submission.csv'
    sample_submission.to_csv(fn, index=False)


    combi = np.mean([preds_logistic_test,preds_xgb_test],axis= 0)
    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    # sample_submission[LIST_CLASSES] = res
    # sample_submission[LIST_CLASSES] = preds_xgb
    sample_submission[LIST_CLASSES] = combi
    if not os.path.exists(fn_out):
        os.mkdir(fn_out)

    # fn = fn_out + 'cat_boost_submission.csv'
    # fn = fn_out + 'xgb_blend_submission.csv'
    fn = fn_out + 'stacked_mean_of_xgb_logistic_submission.csv'
    sample_submission.to_csv(fn, index=False)
print('----------ROC AUC------------')
#print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))
print('using mean %s' %roc_auc_score(Y_valid,np.mean(X_valid,axis=1)))
#print('using catboost %s' %roc_auc_score(Y_valid,preds_cat))
print('using xgb %s' %roc_auc_score(Y_valid,preds_xgb))
print('using lr %s' %roc_auc_score(Y_valid,preds_logistic))
print('using nn %s' %roc_auc_score(Y_valid,preds_nn))
print('using xgb + lr %s' %roc_auc_score(Y_valid,np.mean([preds_logistic,preds_xgb],axis= 0)))
print('using xgb + lr + nn %s' %roc_auc_score(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn],axis= 0)))
print('----------logloss------------')
print('using mean %s' %logloss(Y_valid,np.mean(X_valid,axis=1)))
#print('using catboost %s' %logloss(Y_valid,preds_cat))
print('using xgb %s' %logloss(Y_valid,preds_xgb))
print('using lr %s' %logloss(Y_valid,preds_logistic))
print('using nn %s' %logloss(Y_valid,preds_nn))
print('using xgb + lr %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb],axis= 0)))
print('using xgb + lr + nn %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn],axis= 0)))




if 'tf_nn' in classifiers:


    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():

        x = tf.placeholder(shape=(None,len(csvs_train),len(LIST_CLASSES)), dtype=tf.float32)
        y = tf.placeholder(shape=(None,6),dtype=tf.float32)

        h1 = layers.flatten(x)
        h1 = layers.fully_connected(h1, 64, activation_fn=tf.nn.elu)
        #h2 = layers.fully_connected(h1, 16, activation_fn=tf.nn.elu)
        logits = layers.fully_connected(h1,6,activation_fn=tf.nn.sigmoid)

        cost = tf.losses.log_loss(predictions=logits, labels=y)
        loss = binary_crossentropy(y, logits)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

    epochs = 140
    bsize = 1024
    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(graph=graph,config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):
            step = 0
            while step * bsize < len(X_train):
                batch_x = X_train[step * bsize:(step + 1) * bsize]
                batch_y = Y_train[step * bsize:(step + 1) * bsize]
                _ , logloss_ = sess.run([optimizer,cost], feed_dict={x:batch_x,y:batch_y})
                step += 1

            logloss_val, preds_nn = sess.run([cost,logits], feed_dict={x: X_valid, y: Y_valid})
            print(logloss_val)

        print('using nn %s' % roc_auc_score(Y_valid, preds_nn))


        if do_prediction:
            dfs_test = [pd.read_csv(csv) for csv in csvs_test]
            xs_test = [df[LIST_CLASSES].values for df in dfs_test]
            n_models = len(csvs_test)

            print('Corr matrix')
            print(corr_matrix(xs_test))
            print(' ')

            X_test = X = np.hstack(xs_test)

            step = 0
            res = np.zeros((len(X_test),6))
            while step * bsize < len(X_test):
                batch_x = X_test[step * bsize:(step + 1) * bsize]
                logits_ = sess.run(logits, feed_dict={x:batch_x})


                res[step * bsize:(step + 1) * bsize] = logits_
                step += 1
            sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
            sample_submission[LIST_CLASSES] = res
            if not os.path.exists(fn_out):
                os.mkdir(fn_out)

            fn = fn_out + 'submission.csv'
            sample_submission.to_csv(fn, index=False)






