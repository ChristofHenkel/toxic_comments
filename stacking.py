import numpy as np
from sklearn.preprocessing import minmax_scale
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#import lightgbm as lgb
from xgboost import XGBRegressor
from hyperoptim import do_hyperopt
from utilities import corr_matrix, logloss
from global_variables import LIST_LOGITS, LIST_CLASSES

#classifiers = ['xgb','lr','nn','lgb'] #nn, catboost
#classifiers = ['xgb','nn']
classifiers = ['xgb']
fold_count = 1
do_prediction = False

fn_out = 'models/STACKS/s9/'
# Todo add switch for per class or all logits

models = [#'NBSVM/word_model1/',
          #    'NBSVM/char_model1/',
          #    'RNN/gru_ATT_4/',
          #    'RNN/gru_ATT_6_glove/',
          #    'RNN/gru_CNN/',
          #    'XGB/xgb1/',
          #    'LGB/hurford2/',
          #    'CRNN/cudnn_rnn3/',
          #    'CAPS/cudrnn_caps/',
          #    'CNN/inception5b/',
          #    'RNN/HAN_2/',
          #    'RNN/HLSTM/',
          #      #'CAPS/CAPSNET/',
          #    'CAPS/CAPS3/',
          #      #'CAPS/CAPS2/',
          #    'CAPS/CAPS4/',
          #'PUBLIC/RIDGE/',
          #'CNN/DPCNN/',
          #'CNN/DPCNN_3216/',
          'HANS/gru_fix11/',
          'HANS/mengye_dpcnn/',
          'HANS/text_rcnn/'
          ]

csvs_train = ['models/' + m + 'l2_train_data.csv' for m in models]
csvs_test = ['models/' + m + 'l2_test_data.csv' for m in models]


def get_values(csv_files, columns, hstack = False, with_labels=True, scale = True):
    dfs = [pd.read_csv(csv) for csv in csv_files]
    if scale:
        for label in columns:
            for df in dfs:
                df[label] = np.clip(df[label].values,0,1)
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

print('Corr matrix')
print(corr_matrix(list(X.transpose([1, 0, 2]))))
print(' ')


if 'ho' in classifiers:
    ws = do_hyperopt(csvs_train)
    test_predicts = np.zeros(X[:,0,:].shape)
    for m in range(7):
        test_predicts += ws[m] * X[:,m,:]
    test_predicts /= 7
    print('roc %s logloss %s'%(roc_auc_score(Y,test_predicts),logloss(Y,test_predicts)))



for m in range(len(models)):
    print('%s roc %s logloss %s'%(models[m],roc_auc_score(Y,X[:,m,:]),logloss(Y,X[:,m,:])))



#X_valid, Y_valid = get_values(csvs_valid,columns=LIST_LOGITS,hstack=False,with_labels=True)
X_test = get_values(csvs_test,columns=LIST_CLASSES,hstack=False,with_labels=False)


test_results_list = []
valid_results_list = []

fold_size = len(X) // 10
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == 9:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])


    if 'xgb' in classifiers:
        xgb_clfs = []
        for i in range(6):
            print('fitting xgb on %s' %LIST_CLASSES[i])
            clf = XGBRegressor(objective='reg:logistic', max_depth=2, n_estimators=200, learning_rate=0.1, subsample=0.8,
                               min_child_weight=3)
            clf.fit(X_train[:, :, i], Y_train[:, i])
            xgb_clfs.append(clf)

    if 'lgb' in classifiers:


        #split = len(X_train)//10
        #X_train_lgb = X_train[split:]
        #X_valid_lgb = X_train[:split]
        #Y_train_lgb = Y_train[split:]
        #Y_valid_lgb = Y_train[:split]

        lgb_clfs = []
        for i in range(6):
            print('fitting lgb on %s' % LIST_CLASSES[i])
            lgb_train = lgb.Dataset(X_train[:, :, i], Y_train[:, i])
            lgb_eval = lgb.Dataset(X_valid[:, :, i], Y_valid[:, i], reference=lgb_train)

            params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      #'boosting_type': 'dart',
                      'objective': 'regression',
                      'metric': {'l2', 'auc'},
                      'tree_learner': 'voting',
                      #'num_leaves': 31,
                      'num_leaves': 31,
                      'max_depth':3,
                      'reg_lambda' : 0.2,
                      #'learning_rate': 0.05,
                      'learning_rate': 0.1,
                      'feature_fraction': 0.9,
                      'bagging_fraction': 0.8,
                      'bagging_freq': 5,
                      'verbose': 1,
                      }

            gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=5)
            lgb_clfs.append(gbm)


    if 'lr' in classifiers:
        lr_clfs = []
        for i in range(6):
            print('fitting logistic regression on %s' % LIST_CLASSES[i])
            clf_lr = LogisticRegression()
            clf_lr.fit(X_train[:, :, i], Y_train[:, i])
            lr_clfs.append(clf_lr)

    if 'nn' in classifiers:
        clf_nn = MLPClassifier(solver='adam', batch_size=256, hidden_layer_sizes=(64), max_iter=200, verbose=True,
                               tol=0.000001,)
        clf_nn.loss = 'log_loss'
        X_train2 = X_train.reshape((-1, X_train.shape[1] * len(LIST_CLASSES)))
        clf_nn.fit(X_train2, Y_train)



    preds_cat = np.zeros((len(X_valid), 6))
    preds_xgb = np.zeros((len(X_valid), 6))
    preds_lgb = np.zeros((len(X_valid), 6))
    preds_logistic = np.zeros((len(X_valid), 6))
    preds_nn = np.zeros((len(X_valid), 6))

    combi_valid = []

    if 'xgb' in classifiers:
        for i in range(6):
            preds_xgb[:, i] = xgb_clfs[i].predict(X_valid[:, :, i])
        combi_valid.append(preds_xgb)
    if 'lr' in classifiers:
        for i in range(6):
            preds_logistic[:, i] = lr_clfs[i].predict_proba(X_valid[:, :, i])[:, 1]
        combi_valid.append(preds_logistic)
    if 'lgb' in classifiers:
        for i in range(6):
            preds_lgb[:, i] = lgb_clfs[i].predict(X_valid[:, :, i], num_iteration=lgb_clfs[i].best_iteration)
        combi_valid.append(preds_lgb)
    if 'nn' in classifiers:
        preds_nn = clf_nn.predict_proba(X_valid.reshape((-1, X_valid.shape[1] * len(LIST_CLASSES))))
        combi_valid.append(preds_nn)
    preds_valid = np.mean(combi_valid, axis=0)
    valid_results_list.append(preds_valid)
    if do_prediction:

        combi_test = []

        if 'xgb' in classifiers:
            preds_xgb_test = np.zeros((len(X_test), 6))
            for i in range(6):
                preds_xgb_test[:, i] = xgb_clfs[i].predict(X_test[:, :, i])
            combi_test.append(preds_xgb_test)
        if 'lr' in classifiers:
            preds_logistic_test = np.zeros((len(X_test), 6))
            for i in range(6):
                preds_logistic_test[:, i] = lr_clfs[i].predict_proba(X_test[:, :, i])[:, 1]
            combi_test.append(preds_logistic_test)
        if 'lgb' in classifiers:
            preds_lgb_test = np.zeros((len(X_test), 6))
            for i in range(6):
                preds_lgb_test[:, i] = lgb_clfs[i].predict(X_test[:, :, i], num_iteration=lgb_clfs[i].best_iteration)
            combi_test.append(preds_lgb_test)

        if 'nn' in classifiers:
            preds_nn_test = np.zeros((len(X_test), 6))
            preds_nn_test = clf_nn.predict_proba(X_test.reshape((-1, len(csvs_train) * len(LIST_CLASSES))))
            combi_test.append(preds_nn_test)

        result = np.mean(combi_test, axis= 0)
        test_results_list.append(result)


    print('----------ROC AUC------------')
    print('using mean %s' %roc_auc_score(Y_valid,np.mean(X_valid,axis=1)))
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
    print('using xgb %s' %logloss(Y_valid,preds_xgb))
    print('using lgb %s' %logloss(Y_valid,preds_lgb))
    print('using lr %s' %logloss(Y_valid,preds_logistic))
    print('using nn %s' %logloss(Y_valid,preds_nn))
    print('using xgb + nn %s' %logloss(Y_valid,np.mean([preds_xgb, preds_nn],axis= 0)))
    print('using xgb + lr %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb],axis= 0)))
    print('using xgb + lr + nn %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn],axis= 0)))
    print('using xgb + lr + nn + lgb %s' %logloss(Y_valid,np.mean([preds_logistic,preds_xgb, preds_nn, preds_lgb],axis= 0)))

test_predicts = np.ones(test_results_list[0].shape)
for fold_predict in test_results_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_results_list))


sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
sample_submission[LIST_CLASSES] = test_predicts
fn = fn_out + 'l3_test_data_nn_xgb.csv'
sample_submission.to_csv(fn, index=False)

l3_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
values = np.concatenate(valid_results_list, axis = 0)
l3_data[LIST_LOGITS] = pd.DataFrame(values)
l3_data[LIST_CLASSES] = pd.DataFrame(Y)
l3_data.to_csv(fn_out + 'l3_train_data_nn_xgb.csv')
print('roc %s logloss %s'%(roc_auc_score(Y,values),logloss(Y,values)))