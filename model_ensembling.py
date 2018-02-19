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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.model_selection import KFold
import scipy
from utilities import corr_matrix
from global_variables import LIST_LOGITS, LIST_CLASSES

classification_method = 'catboost' #nn, catboost
do_prediction = True
fn_out = 'models/ENSAMBLES/e2/'

csvs_train = ['models/CNN/inception2_slim/inception2_slim_baggin_logits_folded.csv',
              'models/NBSVM/slim/nbsvm_prediction_valid.csv',
              #'models/RNN/pavel_attention_slim2/baggin_logits_folded.csv',
              'models/RNN/pavel_all_outs_slim/birnn_all_outs_slim_baggin_logits_folded.csv'
              ]

csvs_test = ['models/CNN/inception2_slim/inception2_slim.csv',
             'models/NBSVM/slim/nbsvm_submission.csv',
             #'models/RNN/pavel_attention_slim2/test_data_folded.csv',
             'models/RNN/pavel_all_outs_slim/birnn_all_outs.csv']

dfs = [pd.read_csv(csv) for csv in csvs_train]
xs = [df[LIST_LOGITS].values for df in dfs]
n_models = len(csvs_train)

print('Corr matrix')
print(corr_matrix(xs))
print(' ')


for df in dfs:
    print(roc_auc_score(y_true=df[LIST_CLASSES].values, y_score=df[LIST_LOGITS].values))

def lloss(y_true,y_pred):
    l = 0
    for i in range(6):
        l += log_loss(y_true=y_true[:,i],y_pred=y_pred[:,i])
        l /= 6
    return l

ys = [df[LIST_CLASSES].values for df in dfs]

for i,_ in enumerate(csvs_train[1:]):
    assert np.array_equal(ys[0],ys[i])


Y = ys[0]
X = np.hstack(xs)


if classification_method == 'catbost':
    X = np.concatenate([xs])
    X = X.transpose([1, 0, 2])
split_at = len(X)//10

#kf = KFold(n_splits=10)
#for train, valid in kf.split(X):
#    X_train = X[train]
#    Y_train = Y[train]
#    X_valid = X[valid]
#    Y_valid = Y[valid]

X_train = X[split_at:]
Y_train = Y[split_at:]
X_valid = X[:split_at]
Y_valid = Y[:split_at]

#w = np.zeros((4, 6))
preds = np.zeros((len(X_valid), 6))
#for i in range(6):
#    clf = RandomForestRegressor(n_estimators=1000, verbose=True,n_jobs=10,warm_start=True)
#    clf.fit(X_train[:,:,i],Y_train[:,i])
#    pred = clf.predict(X_valid[:,:,i])
#    preds[:,i] = pred
#    w[:,i] = clf.feature_importances_


if classification_method == 'catboost':
    if do_prediction:
        dfs_test = [pd.read_csv(csv) for csv in csvs_test]
        xs_test = [df[LIST_CLASSES].values for df in dfs_test]
        n_models = len(csvs_test)

        print('Corr matrix')
        print(corr_matrix(xs_test))
        print(' ')

        X_test = X = np.hstack(xs_test)
        res = np.zeros((len(X_test),6))

    for i in range(6):
        model = CatBoostClassifier(iterations=500,learning_rate=0.02, depth=2, loss_function='Logloss',task_type = 'CPU',logging_level='Info')
        #model = LogisticRegression(C=4.0)
        fit_model = model.fit(X_train, Y_train[:,i])
        preds[:,i] = fit_model.predict_proba(X_valid)[:,1]

        if do_prediction:
            res[:, i] = fit_model.predict_proba(X_test)[:, 1]
    print('using catboost %s' % lloss(Y_valid, preds))
    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    sample_submission[LIST_CLASSES] = res
    if not os.path.exists(fn_out):
        os.mkdir(fn_out)

    fn = fn_out + 'cat_boost_submission.csv'
    sample_submission.to_csv(fn, index=False)


print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))

print('using catboost %s' %roc_auc_score(Y_valid,preds))

print('using mean %s' %lloss(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))
print('using catboost %s' %lloss(Y_valid,preds))

tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default():

    x = tf.placeholder(shape=(None,n_models*len(LIST_CLASSES)), dtype=tf.float32)
    y = tf.placeholder(shape=(None,6),dtype=tf.float32)

    h1 = layers.fully_connected(x, 64, activation_fn=tf.nn.elu)
    #h2 = layers.fully_connected(h1, 16, activation_fn=tf.nn.elu)
    logits = layers.fully_connected(h1,6,activation_fn=tf.nn.sigmoid)

    cost = tf.losses.log_loss(predictions=logits, labels=y)
    loss = binary_crossentropy(y, logits)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,momentum=0.9).minimize(loss)

epochs = 70
bsize = 1024
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
with tf.Session(graph=graph,config=config) as sess:
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



for x in xs:
    print(lloss(y_true=Y_valid,y_pred=x[:split_at]))

print('------------->')
print(lloss(y_true=Y_valid,y_pred=logits_val))
print('using mean %s' %lloss(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))
print('using geomean %s' % lloss(Y_valid, scipy.stats.gmean([x[:split_at] for x in xs], axis=0)))
print('----------------------------')
for x in xs:
    print(roc_auc_score(y_true=Y_valid,y_score=x[:split_at]))
print('------------->')
print(roc_auc_score(y_true=Y_valid,y_score=logits_val))
print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))
print('using geomean %s' % roc_auc_score(Y_valid, scipy.stats.gmean([x[:split_at] for x in xs], axis=0)))



