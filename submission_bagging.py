import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np
from utilities import corr_matrix

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

fp = 'submissions/ensembles/one/'
input_fp = fp + 'input/'
new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
csv_files = os.listdir(input_fp)

csv_files = ['models/CNN/inception2_slim/test_data_folded.csv',
             'models/NBSVM/slim/nbsvm_submission.csv',
             'models/RNN/pavel_attention_slim2/test_data_folded.csv',
             'models/RNN/pavel_all_outs_slim/test_data_folded.csv']


test_predicts_list = []
for csv_file in csv_files:
    orig_submission = pd.read_csv(csv_file)
    predictions = orig_submission[list_classes]
    test_predicts_list.append(predictions)

corr_matrix([p.values for p in test_predicts_list])

def bag_by_average(test_predicts_list):
    bagged_predicts = np.ones(test_predicts_list[0].shape)
    for predict in test_predicts_list:
        bagged_predicts += predict

    bagged_predicts/= len(test_predicts_list)
    return  bagged_predicts

def bag_by_geomean(test_predicts_list):
    bagged_predicts = np.ones(test_predicts_list[0].shape)
    for predict in test_predicts_list:
        bagged_predicts *= predict

    bagged_predicts **= (1. / len(test_predicts_list))
    return  bagged_predicts


def bag_by_rank_mean(test_predicts_list):
    p4 = np.vstack([p.values for p in test_predicts_list])
    order = p4.argsort(axis=0)
    ranks = order.argsort(axis=0)
    ranks = np.divide(ranks,ranks.shape[0])
    length = test_predicts_list[0].shape[0]
    #r2 = np.stack([ranks[:length,:],ranks[length:2*length,:],ranks[2*length:,:]], axis=2)
    r2 = np.stack([ranks[i*length:(i+1)*length,:] for i,_ in enumerate(test_predicts_list)], axis=2)
    bagged_ranks = np.mean(r2,axis=2)

    bagged_predicts = np.zeros(bagged_ranks.shape)
    for i in range(6):
        interp = interp1d(ranks[:,i],p4[:,i])
        bagged_predicts[:,i] = interp(bagged_ranks[:,i])
    return bagged_predicts


submission = new_submission.copy()
submission[list_classes] = bag_by_average(test_predicts_list)
submission.to_csv("bag_by_mean.csv", index=False)

submission = new_submission.copy()
submission[list_classes] = bag_by_rank_mean(test_predicts_list)
submission.to_csv(fp + "bag_by_rank_mean.csv", index=False)

submission = new_submission.copy()
submission[list_classes] = bag_by_geomean(test_predicts_list)
submission.to_csv(fp + "bag_by_geomean.csv", index=False)