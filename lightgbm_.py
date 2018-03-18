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

from scipy.sparse import csr_matrix, hstack
from global_variables import LIST_CLASSES, TRAIN_FILENAME, TEST_FILENAME, TRAIN_SLIM_FILENAME, \
    NAN_WORD, COMMENT, SAMPLE_SUBMISSION_FILENAME, LIST_LOGITS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from utilities import write_config


class Config:

    train_fn = TRAIN_FILENAME
    root = 'models/LGB/'
    model_name = 'hurford4/'
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
    lr_solver = 'sag'
    sfm_threshold = 0.2
    lgb_params =  {'learning_rate': 0.2,
                  'application': 'binary',
                  'num_leaves': 31,
                  'verbosity': -1,
                  'metric': {'l2', 'auc'},
                  'data_random_seed': 2,
                  'bagging_fraction': 0.8,
                  'feature_fraction': 0.6,
                  'bagging_freq': 5,
                  'nthread': 4,
                  'lambda_l1': 1,
                  'lambda_l2': 1,
                  'early_stopping_rounds' : 10}
    lgb_rounds_lookup = {'toxic': 140,
                     'severe_toxic': 50,
                     'obscene': 80,
                     'threat': 80,
                     'insult': 70,
                     'identity_hate': 80}

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
for class_name in LIST_CLASSES:
    print(class_name)
    train_target = train[class_name]
    model = LogisticRegression(solver=cfg.lr_solver)
    sfm = SelectFromModel(model, threshold=cfg.sfm_threshold)
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
        d_train = lgb.Dataset(X_train, label=Y_train)
        d_valid = lgb.Dataset(X_valid, label=Y_valid)
        params = cfg.lgb_params
        rounds_lookup = cfg.lgb_rounds_lookup
        print('training lgb')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=rounds_lookup[class_name],
                          valid_sets=d_valid,
                          verbose_eval=10)
        preds_test = model.predict(test_sparse_matrix,num_iteration=model.best_iteration)
        preds_valid = model.predict(X_valid,num_iteration=model.best_iteration)
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