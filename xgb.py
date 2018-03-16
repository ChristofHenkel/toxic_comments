# The goal of this kernel is to demonstrate that LightGBM can have predictive
# performance in line with that of a logistic regression. The theory is that
# labeling is being driven by a few keywords that can be picked up by trees.
#
# With some careful tuning, patience with runtimes, and additional feature
# engineering, this kernel can be tuned to slightly exceed the best
# logistic regression. Best of all, the two approaches (LR and LGB) blend
# well together.
#
# Hopefully, with some work, this could be a good addition to your ensemble.

import gc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import csr_matrix, hstack
from global_variables import LIST_CLASSES, TRAIN_FILENAME, TEST_FILENAME, TRAIN_SLIM_FILENAME, \
    NAN_WORD, COMMENT, SAMPLE_SUBMISSION_FILENAME, LIST_LOGITS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from utilities import write_config


class Config:

    train_fn = TRAIN_FILENAME
    root = 'models/XGB/'
    model_name = 'xgb1/'
    fp = root + model_name
    fn_out_train = 'l2_train_data.csv'
    fp_out_train = fp + fn_out_train
    fn_out_test = 'l2_test_data.csv'
    fp_out_test = fp + fn_out_test
    w_sublinear_tf=True
    w_lowercase=True
    w_strip_accents='unicode'
    w_analyzer='word'
    w_token_pattern=r'\w{1,}'
    w_ngram_range=(1, 2)
    w_max_features=200000
    c_sublinear_tf=True
    c_lowercase=False
    c_strip_accents='unicode'
    c_analyzer='char'
    c_stop_words='english'
    c_ngram_range=(2, 6)
    c_max_features=50000
    objective = 'binary:logistic'
    eta = 0.095
    max_depth = 5
    silent = 1
    eval_metric = 'auc'
    min_child_weight = 2
    subsample = 0.7
    colsample_bytree = 0.7


cfg = Config()
write_config(Config)

train = pd.read_csv(cfg.train_fn).fillna(NAN_WORD)
test = pd.read_csv(TEST_FILENAME).fillna(NAN_WORD)
print('Loaded')

train_text = train[COMMENT]
test_text = test[COMMENT]


word_vectorizer = TfidfVectorizer(
    sublinear_tf=cfg.w_sublinear_tf,
    lowercase=cfg.w_lowercase,
    strip_accents=cfg.w_strip_accents,
    analyzer=cfg.w_analyzer,
    token_pattern=cfg.w_token_pattern,
    ngram_range=cfg.w_ngram_range,
    max_features=cfg.w_max_features)

train_word_features = word_vectorizer.fit_transform(train_text)
print('Word TFIDF 1/2')
test_word_features = word_vectorizer.transform(test_text)
print('Word TFIDF 2/2')

char_vectorizer = TfidfVectorizer(
    sublinear_tf=cfg.c_sublinear_tf,
    lowercase=cfg.c_lowercase,
    strip_accents=cfg.c_strip_accents,
    analyzer=cfg.c_analyzer,
    stop_words=cfg.c_stop_words,
    ngram_range=cfg.c_ngram_range,
    max_features=cfg.c_max_features)

train_char_features = char_vectorizer.fit_transform(train_text)
print('Char TFIDF 1/2')
test_char_features = char_vectorizer.transform(test_text)
print('Char TFIDF 2/2')

train_features = hstack([train_char_features, train_word_features])
print('HStack 1/2')
test_features = hstack([test_char_features, test_word_features])
print('HStack 2/2')

train.drop('comment_text', axis=1, inplace=True)
del test
del train_text
del test_text
del train_char_features
del test_char_features
del train_word_features
del test_word_features
gc.collect()

list_of_preds_test = []
list_of_preds_valid = []
list_of_Y = []
for i, class_name in enumerate(LIST_CLASSES):
    print(class_name)
    train_target = train[class_name]
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=0.2)
    print(train_features.shape)
    print('fitting select from model')
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    test_sparse_matrix = sfm.transform(test_features)
    fold_size = train_sparse_matrix.shape[0] // 10
    kf = KFold(n_splits=10)
    for train_index, valid_index in kf.split(train_sparse_matrix):

        X_train, X_valid = train_sparse_matrix[train_index], train_sparse_matrix[valid_index]
        Y_train, Y_valid = train_target[train_index], train_target[valid_index]
        list_of_Y.append(Y_valid)

        print('training xgb')
        model = XGBRegressor(objective=cfg.objective, max_depth=cfg.max_depth, n_estimators=100, learning_rate=0.1, subsample=cfg.subsample,
                           min_child_weight=cfg.min_child_weight,eta=0.095,colsample_bytree=0.7,silent=False)
        model.fit(X_train, Y_train)
        preds_test = model.predict(test_sparse_matrix)
        preds_valid = model.predict(X_valid)
        list_of_preds_test.append(preds_test)
        list_of_preds_valid.append(preds_valid)

preds_test_list2 = [np.array([list_of_preds_test[l*10+k] for l in range(6)]).T for k in range(10)]
test_predicts = np.ones(preds_test_list2[0].shape)
for fold_predict in preds_test_list2:
    test_predicts *= fold_predict

test_predicts **= (1. / len(preds_test_list2))
new_submission = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)
new_submission[LIST_CLASSES] = test_predicts
new_submission.to_csv(cfg.fp_out_test, index=False)

preds_valid2 = np.array([np.concatenate(list_of_preds_valid[i*10:(i+1)*10]) for i in range(6)]).T
res_y = np.array([np.concatenate(list_of_Y[i*10:(i+1)*10]) for i in range(6)]).T

l2_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
l2_data[LIST_LOGITS] = pd.DataFrame(preds_valid2)
l2_data[LIST_CLASSES] = pd.DataFrame(res_y)
l2_data.to_csv(cfg.fp_out_train)