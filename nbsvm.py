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
from global_variables import TEST_FILENAME, TRAIN_FILENAME, SAMPLE_SUBMISSION_FILENAME

tokenizer = TweetTokenizer()

PREPROCESS = True
FN_OUT_TRAIN = 'models/NBSVM/slim/nbsvm_prediction_train.csv'
FN_OUT_TEST = 'models/NBSVM/slim/nbsvm_prediction_test.csv'
MODE = 'valid'
COMMENT = 'comment_text'

train = pd.read_csv(TRAIN_FILENAME)
#test = pd.read_csv('assets/raw_data/bagging_valid.csv')
test = pd.read_csv(TEST_FILENAME)
subm = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
list_logits = ['logits_' + c for c in label_cols]
train['none'] = 1-train[label_cols].max(axis=1)


train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
train_chars = train.copy()
test_chars = test.copy()

def preprocess(data):

    print('preprocessing')
    p = Preprocessor()
    data[COMMENT] = data[COMMENT].map(lambda x: p.lower(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_breaks(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.expand_contractions(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_ip(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_links_text(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_numbers(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_bigrams(x))
    data[COMMENT] = data[COMMENT].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    return data





if MODE == 'valid':
    train, valid = train_test_split(train, test_size=0.2, random_state=43)
    train_chars, valid_chars = train_test_split(train_chars, test_size=0.2, random_state=43)

if PREPROCESS:
    train = preprocess(train)
    test = preprocess(test)

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
elif MODE == 'test':
    print('Transforming test Tfidf')
    test_word_features = vec.transform(test[COMMENT])
    test_char_features = char_vectorizer.transform(test_chars[COMMENT])
    preds = np.zeros((len(test), len(label_cols)))
else:
    preds = np.zeros((len(train), len(label_cols)))


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
    elif MODE == 'test':
        v = hstack([test_word_features.multiply(r2), test_char_features.multiply(r1)])
    else:
        v = hstack([train_word_features.multiply(r2), train_char_features.multiply(r1)])

    preds[:,i] = m.predict_proba(v)[:,1]

if MODE == 'valid':
    y_true = valid[label_cols].values
    roc = roc_auc_score(y_true, preds)
    print(roc)

elif MODE == 'test':
    try:
        a = test['toxic']
        submission = test_chars.copy()
        submission.drop(columns=["comment_text"])
        submission[list_logits] = pd.DataFrame(preds, index=submission.index)
        submission.to_csv(FN_OUT_TEST, index=False)
    except KeyError:
        submid = pd.DataFrame({'id': subm["id"]})
        submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
        submission.to_csv(FN_OUT_TEST, index=False)

else:
    submission = train_chars.copy()
    submission.drop(columns=["comment_text"])
    submission[list_logits] = pd.DataFrame(preds, index=submission.index)
    submission.to_csv(FN_OUT_TRAIN, index=False)