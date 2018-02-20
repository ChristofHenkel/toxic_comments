import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from textblob import TextBlob
from global_variables import TRAIN_FILENAME, LIST_CLASSES, UNKNOWN_WORD, COMMENT, TEST_FILENAME

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)


Y = train[LIST_CLASSES]
tid = test['id'].values

train['polarity'] = train[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
test['polarity'] = test[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))

train[COMMENT] = train.apply(lambda r: str(r[COMMENT]) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
test[COMMENT] = test.apply(lambda r: str(r[COMMENT]) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

df = pd.concat([train[COMMENT], test[COMMENT]], axis=0)
df = df.fillna(UNKNOWN_WORD)
nrow = train.shape[0]

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=800000)

X = tfidf.transform(train[COMMENT])

model = ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(X, Y)
print(1 - model.score(X, Y))
preds = model.predict_proba(X)
preds = pd.DataFrame([[c[1] for c in preds[row]] for row in range(len(preds))]).T
preds.columns = LIST_CLASSES
preds['id'] = tid
for c in LIST_CLASSES:
    preds[c] = preds[c].clip(0 + 1e12, 1 - 1e12)

