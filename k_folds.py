import numpy as np
import pandas as pd
import os
from operator import itemgetter

fname = 'pavel37'
fp = 'submissions/' + fname + '/'

fns = os.listdir(fp)

fns = [[fn,int(fn.split('k')[1][0]),float(fn.split('v')[1].split('t')[0])] for fn in fns]

csv_files = []
for k in range(10):
    fold_fns = [fn for fn in fns if fn[1] == k]
    best = sorted(fold_fns,key = itemgetter(2),reverse=True)[0]
    csv_files.append(best)

print(np.mean([fn[2] for fn in csv_files]))
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

test_predicts_list = []
for csv_file in csv_files:
    orig_submission = pd.read_csv(fp + csv_file[0])
    predictions = orig_submission[list_classes]
    test_predicts_list.append(predictions)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

#test_predicts = np.multiply(*test_predicts_list)
test_predicts **= (1. / len(test_predicts_list))

new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
new_submission[list_classes] = test_predicts
new_submission.to_csv("fp" + "folded.csv", index=False)