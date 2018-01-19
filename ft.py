from utils import *
from gensim.models.fasttext import FastText
from random import shuffle
import tensorflow as tf
from tensorflow.contrib import layers
import pickle
from sklearn.metrics import log_loss
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

data = read_data(raw_data_dir,train_fn)
model = FastText.load('assets/embedding_models/ft_reviews_dim100_w5_min50/ft_reviews_dim100_w5_min50.ft_model')


data2 = []
for k, d in enumerate(data):
    if k % 100 == 0:
        print(k)
    try:
        vec = model[d['text'].lower()]
    except KeyError:
        vec = model['no word']
    data2.append({'data':vec,'label':d['label']})

shuffle(data2)
split_at = int(len(data)*0.9)
train_data = data2[:split_at]
test_data = data2[split_at:]

with open('train_data.p','wb') as f:
    pickle.dump(train_data,f)
with open('test_data.p','wb') as f:
    pickle.dump(test_data,f)




def calc_log_loss(y_true,y_pred):
    return np.mean([log_loss(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])



with open('train_data.p','rb') as f:
    train_data = pickle.load(f)

with open('test_data.p','rb') as f:
    test_data = pickle.load(f)



X = np.asarray([d['data'] for d in train_data])
Y = np.asarray([d['label'] for d in train_data])
X_test = np.asarray([d['data'] for d in test_data])
Y_test = np.asarray([d['label'] for d in test_data])


clfs = [SVC(kernel='linear', probability=True,verbose=True).fit(X, Y[:, i]) for i in range(Y.shape[1])]
Y_pred = np.array([c.predict_proba(X_test) for c in clfs]).T
print('LinearSVC: %s' %calc_log_loss(Y_test, Y_pred))


clf = RandomForestClassifier(verbose=True).fit(X, Y)
proba = clf.predict_proba(X_test)
p2 = np.asarray(proba)[:,:,1].T
print('RandomForest: %s' %calc_log_loss(Y_test, p2))

import time
n_estimators = 6
start = time.time()
clf3 = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, verbose=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
clf3.fit(X, Y)
end = time.time()
print("Bagging SVC", end - start, clf3.score(X,Y))
proba = clf3.predict_proba(X_test)
p2 = np.asarray(proba)
print('RandomForest: %s' %calc_log_loss(Y_test, p2))


###

test_data = read_data(raw_data_dir,test_fn, mode='test')

model = FastText.load('assets/embedding_models/ft_reviews_dim100_w5_min50/ft_reviews_dim100_w5_min50.ft_model')


data2 = []
for k, d in enumerate(test_data):
    if k % 100 == 0:
        print(k)
    try:
        vec = model[d['text'].lower()]
    except:
        vec = model['no word']
    data2.append({'data':vec,'label':d['label']})

X_submission = np.asarray([d['data'] for d in data2])

with open('X_test.p','wb') as f:
    pickle.dump(X_test,f)

with open('X_test.p','rb') as f:
    X_submission = pickle.load(f)

proba = clf3.predict_proba(X_submission)
#p2 = np.asarray(proba)[:,:,1].T
p2 = np.asarray(proba)

sample_submission = pd.read_csv("assets/raw_data/sample_submission-2.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

p3 = []
for label in list_classes:
    p3.append(p2[:,label2id[label]])
p3 = np.asarray(p3)
sample_submission[list_classes] = p3.T
sample_submission.to_csv("baseline.csv", index=False)