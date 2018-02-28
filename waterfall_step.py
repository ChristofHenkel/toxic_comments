import pandas as pd
import os
from global_variables import LIST_CLASSES, TRAIN_SLIM_FILENAME, LIST_LOGITS, TEST_FILENAME
import numpy as np
from sklearn.metrics import roc_auc_score
from utilities import logloss

predictions = pd.read_csv('models/PUBLIC/one_more_blend 0.9847.csv', index_col=0)
valids = pd.read_csv('models/RNN/pavel_all_outs_slim/birnn_all_outs_slim_baggin_logits_folded.csv', index_col=1)
#a = predictions.loc[predictions[LIST_CLASSES] > 0.99]
test = pd.read_csv(TEST_FILENAME, index_col=0)
def find_good_predicts(valids, toxic_column):

    b1 = valids[toxic_column] > 0.99
    b2 = valids[toxic_column] < 0.001
    c = valids[b1|b2]
    print(valids[b1].shape)
    print(valids[b2].shape)
    return c



c = find_good_predicts(valids, 'logits_toxic')
print(c.shape)

print(logloss(c[LIST_CLASSES].values,c[LIST_LOGITS].values))
print(logloss(valids[LIST_CLASSES].values,valids[LIST_LOGITS].values))

good_predictions = find_good_predicts(predictions,'toxic')
g = good_predictions.join(test)
train = pd.read_csv(TRAIN_SLIM_FILENAME,index_col=1)
train = train.drop(columns=['Unnamed: 0'])
train_e0 = pd.concat([train,g])
train_e0.to_csv('train_e0.csv')


