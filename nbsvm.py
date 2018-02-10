import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer, ToktokTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack
#from sklearn.naive_bayes import MultinomialNB
from preprocess_utils import Preprocessor
#from sklearn.ensemble import BaggingClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import num2words

tokenizer = TweetTokenizer()
PREPROCESS = True
train = pd.read_csv('assets/raw_data/train.csv')
test = pd.read_csv('assets/raw_data/test.csv')
subm = pd.read_csv('assets/raw_data/sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
train_chars = train.copy()

def preprocess(train):
    
    print('preprocessing')
    p = Preprocessor()
    train[COMMENT] = train[COMMENT].map(lambda x: p.lower(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.rm_breaks(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.expand_contractions(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.rm_ip(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.rm_links_text(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.replace_numbers(x))
    train[COMMENT] = train[COMMENT].map(lambda x: p.rm_bigrams(x))
    train[COMMENT] = train[COMMENT].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    return train
[print(train[COMMENT][i]) for i in range(20,30)]


MODE = 'test'

if MODE == 'valid':
    train, valid = train_test_split(train, test_size=0.2, random_state=43)
    train_chars, valid_chars = train_test_split(train_chars, test_size=0.2, random_state=43)

n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2),
                      tokenizer=tokenizer.tokenize,
                      lowercase=True,
                      min_df=3,
                      max_df=0.9,
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1 )

char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  lowercase=False,
                                  strip_accents='unicode',
                                  analyzer='char',
                                  ngram_range=(1, 4),
                                  max_features=30000)
print('fitting word Tfidf')
train_word_features = vec.fit_transform(train[COMMENT])
print('fitting char Tfidf')
train_char_features = char_vectorizer.fit_transform(train_chars[COMMENT])

if MODE == 'valid':
    print('Transforming valid Tfidf')
    valid_word_features = vec.transform(valid[COMMENT])
    valid_char_features = char_vectorizer.transform(valid_chars[COMMENT])
    preds = np.zeros((len(valid), len(label_cols)))
else:
    print('Transforming test Tfidf')
    test_word_features = vec.transform(test[COMMENT])
    test_char_features = char_vectorizer.transform(test[COMMENT])
    preds = np.zeros((len(test), len(label_cols)))

def pr(y_i, y, feature):
    p = feature[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)




for i, j in enumerate(label_cols):
    print('fit', j)
    y = train[j].values
    r1 = np.log(pr(1, y,train_char_features) / pr(0, y,train_char_features))
    r2 = np.log(pr(1, y, train_word_features) / pr(0, y, train_word_features))
    m = LogisticRegression(C=4, dual=True)
    x_nb1 = train_word_features.multiply(r2)
    x_nb2 = train_char_features.multiply(r1)
    x_nb = hstack([x_nb1,x_nb2])
    m.fit(x_nb, y)

    if MODE == 'valid':
        v = hstack([valid_word_features.multiply(r2),valid_char_features.multiply(r1)])
    else:
        v = hstack([test_word_features.multiply(r2), test_char_features.multiply(r1)])

    preds[:,i] = m.predict_proba(v)[:,1]

if MODE == 'valid':
    y_true = valid[label_cols].values
    roc = roc_auc_score(y_true, preds)
    print(roc)

else:
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
    submission.to_csv('models/NBSVM/submission.csv', index=False)