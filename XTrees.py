import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from textblob import TextBlob
from utilities import logloss
from global_variables import TRAIN_FILENAME, LIST_CLASSES, UNKNOWN_WORD, COMMENT, TEST_FILENAME, TRAIN_SLIM_FILENAME, VALID_SLIM_FILENAME

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

train = pd.read_csv(TRAIN_SLIM_FILENAME)
valid = pd.read_csv(VALID_SLIM_FILENAME)
test = pd.read_csv(TEST_FILENAME)


Y_train = train[LIST_CLASSES]
tid = test['id'].values

print('calculating polarity for train')
train['polarity'] = train[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
print('calculating polarity for valid')
valid['polarity'] = valid[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
print('calculating polarity for test')
test['polarity'] = test[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))

print('adding polarity to text train')
train[COMMENT] = train.apply(lambda r: str(r[COMMENT]) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
print('adding polarity to text valid')
valid[COMMENT] = valid.apply(lambda r: str(r[COMMENT]) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
print('adding polarity to text test')
test[COMMENT] = test.apply(lambda r: str(r[COMMENT]) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

#df = pd.concat([train[COMMENT], valid[COMMENT], test[COMMENT]], axis=0)
#df = df.fillna(UNKNOWN_WORD)
#nrow = train.shape[0]

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=800000)

print('transforming train')
_ = tfidf.fit_transform(pd.concat([train[COMMENT], valid[COMMENT], test[COMMENT]], axis=0))


X_train = tfidf.transform(train[COMMENT])
model = ExtraTreesClassifier(n_jobs=-1, random_state=3, verbose=True)
print('fitting classifier')
model.fit(X_train, Y_train)

print('transforming valid')
X_valid = tfidf.transform(valid[COMMENT])

preds = model.predict_proba(X_valid)
Y_valid = valid[LIST_CLASSES].values
a = np.array(preds)[:,:,1].T
print(logloss(Y_valid,a))


#preds = pd.DataFrame([[c[1] for c in preds[row]] for row in range(len(preds))]).T
#preds.columns = LIST_CLASSES
#preds['id'] = tid
#for c in LIST_CLASSES:
#    preds[c] = preds[c].clip(0 + 1e12, 1 - 1e12)

