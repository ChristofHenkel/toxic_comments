import numpy
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from sklearn.metrics import log_loss
import tqdm
from sklearn.preprocessing import minmax_scale
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import scipy
from utilities import corr_matrix, logloss
from global_variables import LIST_LOGITS, LIST_CLASSES

classifiers = ['xgb','lr','nn','lgb'] #nn, catboost
#classifiers = ['xgb','nn']
#classifiers = ['lgb'] #nn, catboost
do_prediction = False
fn_out = 'models/STACKS/s1/'
# Todo add switch for per class or all logits


stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt", learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8, bagging_freq=5, reg_lambda=0.2)

csvs_train = ['models/NBSVM/word_model1/l2_train_data.csv',
              'models/NBSVM/char_model1/l2_train_data.csv',
              'models/RNN/gru_ATT_4/l2_train_data.csv',
              'models/RNN/gru_ATT_6_glove/l2_train_data.csv',
              'models/RNN/gru_CNN/l2_train_data.csv',
              'models/XGB/xgb1/l2_train_data.csv',
              'models/LGB/hurford2/l2_train_data.csv',
              'models/CRNN/cudnn_rnn3/l2_train_data.csv',
              #'models/CNN/vgg5_dilations/l2_train_data.csv',
              'models/CNN/inception5b/l2_train_data.csv',
              ]



csvs_test = ['models/NBSVM/word_model1/l2_test_data.csv',
              'models/NBSVM/char_model1/l2_test_data.csv',
              'models/RNN/gru_ATT_4/l2_test_data.csv',
              'models/RNN/gru_ATT_6_glove/l2_test_data.csv',
              'models/RNN/gru_CNN/l2_test_data.csv',
              'models/XGB/xgb1/l2_test_data.csv',
              'models/LGB/hurford2/l2_test_data.csv',
              'models/CRNN/cudnn_rnn3/l2_test_data.csv',
              #'models/CNN/vgg5_dilations/l2_test_data.csv',
              'models/CNN/inception5b/l2_test_data.csv',
              ]


def get_values(csv_files, columns, hstack = False, with_labels=True, scale = True):
    dfs = [pd.read_csv(csv) for csv in csv_files]
    if scale:
        for label in columns:
            for df in dfs:
                df[label] = minmax_scale(df[label])


    xs = [df[columns].values for df in dfs]
    #xs = [df[columns].values for df in dfs[:2]]
    #xs.append(np.mean([df[columns].values for df in dfs[2:4]],axis=0))
    #xs.extend([df[columns].values for df in dfs[4:]])
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

X, Y = get_values(csvs_train,columns=LIST_LOGITS,hstack=False,with_labels=True)
split = len(X)//10

X_train = X[split:]
X_valid = X[:split]
Y_train = Y[split:]
Y_valid = Y[:split]

#X_valid, Y_valid = get_values(csvs_valid,columns=LIST_LOGITS,hstack=False,with_labels=True)
X_test = get_values(csvs_test,columns=LIST_CLASSES,hstack=False,with_labels=False)


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

if 'lgb' in classifiers:


    split = len(X_train)//10
    X_train_lgb = X_train[split:]
    X_valid_lgb = X_train[:split]
    Y_train_lgb = Y_train[split:]
    Y_valid_lgb = Y_train[:split]



    lgb_clfs = []
    for i in range(6):
        print('fitting lgb on %s' % LIST_CLASSES[i])
        lgb_train = lgb.Dataset(X_train_lgb[:, :, i], Y_train_lgb[:, i])
        lgb_eval = lgb.Dataset(X_valid_lgb[:, :, i], Y_valid_lgb[:, i], reference=lgb_train)

        params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': {'l2', 'auc'},
                  #'num_leaves': 31,
                  'num_leaves': 20,
                  #'learning_rate': 0.05,

                  'learning_rate': 0.1,
                  'feature_fraction': 0.9,
                  'bagging_fraction': 0.8,
                  'bagging_freq': 5,
                  'verbose': 1,
                  }
        gbm = lgb.train(params,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)
        lgb_clfs.append(gbm)

    #preds_lgb = np.zeros((len(X_valid), 6))
    #for i in range(6):
    #    preds_lgb[:, i] = lgb_clfs[i].predict(X_valid[:, :, i], num_iteration=lgb_clfs[i].best_iteration)
    #print('using lgb %s' % roc_auc_score(Y_valid, preds_lgb))
    #print('using lgb %s' %logloss(Y_valid,preds_lgb))
    #print(lr)
    #time.sleep(3)


if 'lr' in classifiers:
    lr_clfs = []
    for i in range(6):
        print('fitting logistic regression on %s' % LIST_CLASSES[i])
        clf_lr = LogisticRegression()
        clf_lr.fit(X_train[:, :, i], Y_train[:, i])
        lr_clfs.append(clf_lr)

if 'nn' in classifiers:
    clf_nn = MLPClassifier(solver='adam', batch_size=256, hidden_layer_sizes=(64,), max_iter=200, verbose=True,
                           tol=0.000001,)
    clf_nn.loss = 'log_loss'
    X_train2 = X_train.reshape((-1, X_train.shape[1] * len(LIST_CLASSES)))
    clf_nn.fit(X_train2, Y_train)



preds_cat = np.zeros((len(X_valid), 6))
preds_xgb = np.zeros((len(X_valid), 6))
preds_lgb = np.zeros((len(X_valid), 6))
preds_logistic = np.zeros((len(X_valid), 6))
preds_nn = np.zeros((len(X_valid), 6))

for i in range(6):
    if 'xgb' in classifiers:
        preds_xgb[:, i] = xgb_clfs[i].predict(X_valid[:, :, i])
    if 'lr' in classifiers:
        preds_logistic[:, i] = lr_clfs[i].predict_proba(X_valid[:, :, i])[:, 1]
    if 'lgb' in classifiers:
        preds_lgb[:, i] = lgb_clfs[i].predict(X_valid[:, :, i])
if 'nn' in classifiers:
    preds_nn = clf_nn.predict_proba(X_valid.reshape((-1, X_valid.shape[1] * len(LIST_CLASSES))))


if do_prediction:

    combi = []

    if 'xgb' in classifiers:
        preds_xgb_test = np.zeros((len(X_test), 6))
        for i in range(6):
            preds_xgb_test[:, i] = xgb_clfs[i].predict(X_test[:, :, i])
        combi.append(preds_xgb_test)
    if 'lr' in classifiers:
        preds_logistic_test = np.zeros((len(X_test), 6))
        for i in range(6):
            preds_logistic_test[:, i] = lr_clfs[i].predict_proba(X_test[:, :, i])[:, 1]
        combi.append(preds_logistic_test)
    if 'lgb' in classifiers:
        preds_lgb_test = np.zeros((len(X_test), 6))
        for i in range(6):
            preds_lgb_test[:, i] = lgb_clfs[i].predict(X_test[:, :, i], num_iteration=lgb_clfs[i].best_iteration)
        combi.append(preds_lgb_test)

    if 'nn' in classifiers:
        preds_nn_test = np.zeros((len(X_test), 6))
        preds_nn_test = clf_nn.predict_proba(X_test.reshape((-1, len(csvs_train) * len(LIST_CLASSES))))
        combi.append(preds_nn_test)

    result = np.mean(combi,axis= 0)
    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    sample_submission[LIST_CLASSES] = result
    if not os.path.exists(fn_out):
        os.mkdir(fn_out)

    fn = fn_out + '_stacked_' + '_'.join(classifiers) + '.csv'
    sample_submission.to_csv(fn, index=False)
print('----------ROC AUC------------')
#print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[:split_at] for x in xs],axis=0)))
print('using mean %s' %roc_auc_score(Y_valid,np.mean(X_valid,axis=1)))
#print('using catboost %s' %roc_auc_score(Y_valid,preds_cat))
print('using xgb %s' %roc_auc_score(Y_valid,preds_xgb))
print('using lgb %s' %roc_auc_score(Y_valid,preds_lgb))
print('using lr %s' %roc_auc_score(Y_valid,preds_logistic))
print('using nn %s' %roc_auc_score(Y_valid,preds_nn))
print('using xgb + lr %s' %roc_auc_score(Y_valid,np.mean([preds_logistic,preds_xgb],axis= 0)))
print('using xgb + nn %s' %roc_auc_score(Y_valid,np.mean([preds_xgb, preds_nn],axis= 0)))
print('using xgb + lr + nn %s' %roc_auc_score(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn],axis= 0)))
print('using xgb + lr + nn + lgb %s' %roc_auc_score(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn, preds_lgb],axis= 0)))
print('----------logloss------------')
print('using mean %s' %logloss(Y_valid,np.mean(X_valid,axis=1)))
#print('using catboost %s' %logloss(Y_valid,preds_cat))
print('using xgb %s' %logloss(Y_valid,preds_xgb))
print('using lgb %s' %logloss(Y_valid,preds_lgb))
print('using lr %s' %logloss(Y_valid,preds_logistic))
print('using nn %s' %logloss(Y_valid,preds_nn))
print('using xgb + nn %s' %logloss(Y_valid,np.mean([preds_xgb, preds_nn],axis= 0)))
print('using xgb + lr %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb],axis= 0)))
print('using xgb + lr + nn %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn],axis= 0)))
print('using xgb + lr + nn + lgb %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn, preds_lgb],axis= 0)))





