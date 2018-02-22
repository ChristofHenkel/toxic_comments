import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer, ToktokTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
from preprocess_utils import Preprocessor
from global_variables import TEST_FILENAME, COMMENT, LIST_CLASSES, LIST_LOGITS
import os

tokenizer = TweetTokenizer()

PREPROCESS = True
FN_OUT_TRAIN = 'models/NBSVM/slim/l2_train_data.csv'

train = pd.read_csv('assets/raw_data/bagging_train.csv')
#test = pd.read_csv('assets/raw_data/bagging_valid.csv')
test = pd.read_csv(TEST_FILENAME)
subm = pd.read_csv('assets/raw_data/sample_submission.csv')
fold_count = 10

train['none'] = 1-train[LIST_CLASSES].max(axis=1)


train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

def preprocess(data):

    print('preprocessing')
    p = Preprocessor()
    data[COMMENT] = data[COMMENT].map(lambda x: p.lower(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_breaks(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.expand_contractions(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_ip(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_links_text(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_numbers(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_bigrams(x))
    data[COMMENT] = data[COMMENT].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    return data


if PREPROCESS:
    train = preprocess(train)
    test = preprocess(test)

def pr(y_i, y, feature):
    p = feature[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


fold_size = train.shape[0] // 10
X = train
Y = train[LIST_CLASSES].values

preds = np.zeros((len(train), len(LIST_CLASSES)))
res_y = np.zeros((len(train), len(LIST_CLASSES)))
for fold_id in range(0, fold_count):

    #reinitialize Vectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2),
                          tokenizer=tokenizer.tokenize,
                          lowercase=True,
                          min_df=3,
                          max_df=0.9,
                          strip_accents='unicode',
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1)


    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_train = pd.concat([X[:fold_start], X[fold_end:]])
    X_valid = X[fold_start:fold_end]

    #Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])
    #Y_valid = Y[fold_start:fold_end]

    print('fitting word Tfidf for fold %s' %fold_id)
    train_word_features = vec.fit_transform(X_train[COMMENT])
    valid_word_features = vec.transform(X_valid[COMMENT])

    for i, j in enumerate(LIST_CLASSES):
        print('fit', j)
        y = X_train[j].values
        r2 = np.log(pr(1, y, train_word_features) / pr(0, y, train_word_features))
        m = LogisticRegression(C=4, dual=True)
        x_nb = train_word_features.multiply(r2)
        m.fit(x_nb, y)
        v = valid_word_features.multiply(r2)
        preds[fold_start:fold_end,i] = m.predict_proba(v)[:,1]
        res_y[fold_start:fold_end,i] = Y[fold_start:fold_end,i]

#submission = train.copy()
#submission.drop(columns=["comment_text"])
#submission[LIST_LOGITS] = pd.DataFrame(preds, index=submission.index)

#submission.to_csv(FN_OUT_TRAIN, index=False)

l2_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
l2_data[LIST_LOGITS] = pd.DataFrame(preds)
l2_data[LIST_CLASSES] = pd.DataFrame(res_y)
l2_data.to_csv(FN_OUT_TRAIN)