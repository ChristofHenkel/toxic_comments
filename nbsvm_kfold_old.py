import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer, ToktokTokenizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from preprocess_utils import Preprocessor, preprocess
from utilities import write_config, logloss
from global_variables import TEST_FILENAME, COMMENT, LIST_CLASSES, LIST_LOGITS, TRAIN_SLIM_FILENAME, \
    SAMPLE_SUBMISSION_FILENAME, VALID_SLIM_FILENAME, NAN_WORD, TRAIN_FILENAME
import os


class Config:

    do_preprocess = False
    root = 'models/NBSVM/'
    model_name = 'word_model1/'
    fp = root + model_name
    fn_out_train = 'l2_train_data.csv'
    fp_out_train = fp + fn_out_train
    fn_out_test = 'l2_test_data.csv'
    fp_out_test = fp + fn_out_test
    train_fn = TRAIN_FILENAME
    fold_count = 10
    levels = ['word']
    tokenizer = TweetTokenizer()
    w_ngram_range = (1, 2)
    w_tokenizer = tokenizer.tokenize
    w_lowercase = True
    w_min_df = 3
    w_max_df = 0.9
    w_strip_accents = 'unicode'
    w_use_idf = 1
    w_smooth_idf = 1
    w_sublinear_tf = 1
    c_sublinear_tf = True
    c_lowercase = False
    c_strip_accents = 'unicode'
    c_analyzer = 'char'
    c_ngram_range = (1, 4)
    c_max_features = 30000

cfg = Config()
write_config(Config)


if not os.path.exists(cfg.root + cfg.model_name):
    os.mkdir(cfg.root + cfg.model_name)

train = pd.read_csv(cfg.train_fn)
test = pd.read_csv(TEST_FILENAME, index_col=0)
subm = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)


train['none'] = 1-train[LIST_CLASSES].max(axis=1)


train[COMMENT].fillna(NAN_WORD, inplace=True)
test[COMMENT].fillna(NAN_WORD, inplace=True)

if cfg.do_preprocess:
    train = preprocess(train)
    test = preprocess(test)

def pr(y_i, y, feature):
    p = feature[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


fold_size = train.shape[0] // cfg.fold_count
X = train[COMMENT]
Y = train[LIST_CLASSES]

preds_test_list = []
preds_valid_list = []
list_of_y = []
preds_valid = np.zeros((len(train), len(LIST_CLASSES)))
res_y = np.zeros((len(train), len(LIST_CLASSES)))
kf = KFold(n_splits=3)
for train_index, valid_index in kf.split(X):
    #a = kf.split(X)
    #train_index, valid_index = a.__next__()
    preds_test_fold_list = []
    preds_valid_fold_list = []

    X_train, X_valid = X[train_index], X[valid_index]
    Y_train, Y_valid = Y.loc[train_index].values, Y.loc[valid_index].values

    #reinitialize Vectorizer

    if 'word' in cfg.levels:
        word_vectorizer = TfidfVectorizer(ngram_range=cfg.w_ngram_range,
                                          tokenizer=cfg.w_tokenizer,
                                          lowercase=cfg.w_lowercase,
                                          min_df=cfg.w_min_df,
                                          max_df=cfg.w_max_df,
                                          strip_accents=cfg.w_strip_accents,
                                          use_idf=cfg.w_use_idf,
                                          smooth_idf=cfg.w_smooth_idf,
                                          sublinear_tf=cfg.w_sublinear_tf)

        #print('fitting word Tfidf for fold %s' %fold_id)
        print('fitting word Tfidf')
        train_word_features = word_vectorizer.fit_transform(X_train)
        valid_word_features = word_vectorizer.transform(X_valid)
        test_word_features = word_vectorizer.transform(test[COMMENT])

    if 'char' in cfg.levels:
        char_vectorizer = TfidfVectorizer(sublinear_tf=cfg.c_sublinear_tf,
                                          lowercase=cfg.c_lowercase,
                                          strip_accents=cfg.c_strip_accents,
                                          analyzer=cfg.c_analyzer,
                                          ngram_range=cfg.c_ngram_range,
                                          max_features=cfg.c_max_features)

        #print('fitting char Tfidf for fold %s' % fold_id)
        print('fitting char Tfidf')
        train_char_features = char_vectorizer.fit_transform(X_train)
        valid_char_features = char_vectorizer.transform(X_valid)
        test_char_features = char_vectorizer.transform(test[COMMENT])

    for i, j in enumerate(LIST_CLASSES):
        print('fit', j)
        y = Y_train[:,i]
        y_valid = Y_valid[:,i]
        if 'char' in cfg.levels:
            r1 = np.log(pr(1, y, train_char_features) / pr(0, y, train_char_features))
            x_nb1 = train_char_features.multiply(r1)
        if 'word' in cfg.levels:
            r2 = np.log(pr(1, y, train_word_features) / pr(0, y, train_word_features))
            x_nb2 = train_word_features.multiply(r2)

        if 'char' in cfg.levels and 'word' in cfg.levels:
            valid_features = hstack([valid_word_features.multiply(r2), valid_char_features.multiply(r1)])
            test_features = hstack([test_word_features.multiply(r2), test_char_features.multiply(r1)])
            x_nb = hstack([x_nb2, x_nb1])
        elif 'char' in cfg.levels:
            x_nb = x_nb1
            valid_features = valid_char_features.multiply(r1)
            test_features = test_char_features.multiply(r1)
        else:
            x_nb = x_nb2
            valid_features = valid_word_features.multiply(r2)
            test_features = test_word_features.multiply(r2)

        m = LogisticRegression(C=4, dual=True)
        m.fit(x_nb, y)

        preds_valid = m.predict_proba(valid_features)[:, 1]
        preds_test = m.predict_proba(test_features)[:, 1]
        preds_test_fold_list.append(preds_test)
        preds_valid_fold_list.append(preds_valid)

    preds_test_list.append(np.array(preds_test_fold_list).T)
    preds_valid_list.append(np.array(preds_valid_fold_list).T)
    list_of_y.append(Y_valid)

    print('logloss: %s'%logloss(Y_valid,np.array(preds_valid_fold_list).T))
    print('ROC: %s' % roc_auc_score(Y_valid,np.array(preds_valid_fold_list).T))

l2_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
l2_data[LIST_LOGITS] = pd.DataFrame(np.concatenate(preds_valid_list, axis = 0))
l2_data[LIST_CLASSES] = pd.DataFrame(np.concatenate(list_of_y, axis = 0))
l2_data.to_csv(cfg.fp_out_train)

preds_test_list2 = [np.array(preds_test_list[i:i+6]).T for i in range(10)]
test_predicts = np.ones(preds_test_list2[0].shape)
for fold_predict in preds_test_list2:
    test_predicts *= fold_predict

test_predicts **= (1. / len(preds_test_list2))
new_submission = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)
new_submission[LIST_CLASSES] = test_predicts
new_submission.to_csv(cfg.fp_out_test, index=False)