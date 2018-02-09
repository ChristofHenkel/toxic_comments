import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

tokenizer = TweetTokenizer()

train = pd.read_csv('assets/raw_data/train.csv')
test = pd.read_csv('assets/raw_data/test.csv')
subm = pd.read_csv('assets/raw_data/sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)



train, valid = train_test_split(train, test_size=0.2)

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenizer.tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_term_doc = vec.fit_transform(train[COMMENT])
valid_term_doc = vec.transform(valid[COMMENT])
#test_term_doc = vec.transform(test[COMMENT])


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=30000)
char_vectorizer.fit(train[COMMENT])
train_char_features = char_vectorizer.transform(train_text)



def pr(y_i, y):
    p = train_term_doc[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


#test_x = test_term_doc


preds = np.zeros((len(valid), len(label_cols)))

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

for i, j in enumerate(label_cols):
    print('fit', j)
    y = train[j].values
    r = np.log(pr(1, y) / pr(0, y))
    m1 = LogisticRegression(C=4, dual=True)
    #m = RandomForestClassifier(max_depth=2, random_state=0)
    #m = SGDClassifier(loss='modified_huber', max_iter=10)
    m2 = MultinomialNB(alpha=0.015)
    #x_nb = train_term_doc.multiply(r)
    m = AdaBoostClassifier(m1)
    m.fit(train_term_doc, y)

    #m.fit(x_nb, y)
    preds[:,i] = m.predict_proba(valid_term_doc.multiply(r))[:,1]


y_true = valid[label_cols].values


roc = roc_auc_score(y_true, preds)
print(roc)

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)

classif = OneVsRestClassifier(LogisticRegression(C=4, dual=True))
classif.fit(X, Y)