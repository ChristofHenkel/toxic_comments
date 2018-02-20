"""
https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/neptune_ensembling.ipynb
"""

import numpy
from catboost import CatBoostRegressor,CatBoostClassifier
import pickle
import numpy as np
import tensorflow as tf
from train_model import ToxicComments
from tqdm import tqdm
import pandas as pd
import os
from utilities import coverage
from sklearn.metrics import roc_auc_score

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_logits = ['logits_' + c for c in list_classes]


csv_1 = 'models/CNN/inception_2_2/train_logits_folded.csv'
df1 = pd.read_csv(csv_1)
p1 = df1[list_logits].values

csv_2 = 'models/RNN/pavel_baseline/train_logits_folded.csv'
df2 = pd.read_csv(csv_2)
p2 = df2[list_logits].values



t1 = df1[list_classes].values
t2 = df2[list_classes].values

a = np.array_equal(t1,t2)

X = np.hstack([p1,p2])
Y = t1

split_at = len(X)//10

X_train = X[split_at:]
Y_train = Y[split_at:]
X_valid = X[:split_at]
Y_valid = Y[:split_at]


for i, label in enumerate(list_classes):
    model = CatBoostClassifier(iterations=500,learning_rate=0.02, depth=2, loss_function='Logloss',task_type = 'GPU',logging_level='Info')
    fit_model = model.fit(X_train, Y_train[:,i])


    Y_preds = fit_model.predict_proba(X_valid)

    print('old1 %s' %roc_auc_score(Y_valid[:,i],p1[:split_at][:,i]))
    print('old2 %s' %roc_auc_score(Y_valid[:,i],p2[:split_at][:,i]))
    print('using mean %s' %roc_auc_score(Y_valid[:,i],np.mean([p1[:split_at][:,i],p2[:split_at][:,i]],axis=0)))
    print('new %s' %roc_auc_score(Y_valid[:,i],Y_preds[:,1]))




def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()

X_valid = np.hstack([bad_word_logreg_valid.drop('id',axis=1),
                     bad_word_count_logreg_valid.drop('id',axis=1),
                     char_vdcnn_valid.drop('id',axis=1),
                     count_logreg_valid.drop('id',axis=1),
                     glove_dpcnn_valid.drop('id',axis=1),
                     glove_lstm_valid.drop('id',axis=1),
                     glove_scnn_valid.drop('id',axis=1),
                     tfidf_logreg_valid.drop('id',axis=1),
                     word_lstm_valid.drop('id',axis=1)])

y_valid_multilabel = labels_valid[LABEL_COLUMNS].values

X_test = np.hstack([bad_word_logreg_test.drop('id',axis=1),
                    bad_word_count_logreg_test.drop('id',axis=1),
                    char_vdcnn_test.drop('id',axis=1),
                    count_logreg_test.drop('id',axis=1),
                    glove_dpcnn_test.drop('id',axis=1),
                    glove_lstm_test.drop('id',axis=1),
                    glove_scnn_test.drop('id',axis=1),
                    tfidf_logreg_test.drop('id',axis=1),
                    word_lstm_test.drop('id',axis=1)])


from sklearn.model_selection import KFold


def fit_cv(X, y, n_splits=10):
    estimators, scores = [], []
    kf = KFold(n_splits=n_splits)
    for train, valid in kf.split(X):
        X_train_ = X[train]
        y_train_ = y[train]
        X_valid_ = X[valid]
        y_valid_ = y[valid]

        estimators_fold = []
        for i in tqdm(range(6)):
            y_train_one_label = y_train_[:, i]
            estimator = CatBoostClassifier(iterations=500,
                                           learning_rate=0.02,
                                           depth=2,
                                           verbose=False)
            estimator.fit(X_train_, y_train_one_label)
            estimators_fold.append(estimator)
        estimators.append(estimators_fold)

        y_valid_pred = []
        for estimator in estimators_fold:
            y_valid_pred_one_label = estimator.predict_proba(X_valid_)
            y_valid_pred.append(y_valid_pred_one_label)
        y_valid_pred = np.stack(y_valid_pred, axis=1)[..., 1]
        score = multi_roc_auc_score(y_valid_, y_valid_pred)
        scores.append(score)
    return scores, estimators


scores, estimators = fit_cv(X_valid, y_valid_multilabel)


print('score average {}\nscore std {}'.format(np.mean(scores), np.std(scores)))

# Ensemble Prediction


y_bagged = []
for estimators_fold in estimators:
    y_test_pred = []
    for estimator in estimators_fold:
        y_test_pred_one_label = estimator.predict_proba(X_test)
        y_test_pred.append(y_test_pred_one_label)
    y_test_pred = np.stack(y_test_pred, axis=1)[..., 1]
    y_bagged.append(y_test_pred)
y_bagged = np.mean(np.stack(y_bagged), axis=0)

