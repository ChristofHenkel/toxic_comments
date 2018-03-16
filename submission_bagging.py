import pandas as pd
import os
from scipy.interpolate import interp1d
import numpy as np
from utilities import corr_matrix
from global_variables import LIST_CLASSES, TEST_FILENAME

fp = 'models/ENSAMBLES/e0/'
add_comments = True
input_fp = fp + 'input/'
if add_comments:
    new_submission = pd.read_csv(TEST_FILENAME)
else:
    new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
#csv_files = os.listdir(input_fp)

#csv_files = ['models/CNN/inception2_slim/l2_test_data.csv',
#             'models/NBSVM/slim/nbsvm_submission.csv',
#             'models/RNN/pavel_attention_slim2/l2_test_data.csv',
#             'models/RNN/pavel_all_outs_slim/l2_test_data.csv']

csv_files = ['models/PUBLIC/' + fn for fn in os.listdir('models/PUBLIC/') if fn.endswith('.csv')]

test_predicts_list = []
for csv_file in csv_files:
    orig_submission = pd.read_csv(csv_file)
    predictions = orig_submission[LIST_CLASSES]
    test_predicts_list.append(predictions)

corr_matrix([p.values for p in test_predicts_list])

def bag_by_average(test_predicts_list):
    bagged_predicts = np.zeros(test_predicts_list[0].shape)
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
if add_comments:
    res = bag_by_average(test_predicts_list)
    for i,label in enumerate(LIST_CLASSES):

        submission[label] = res[:,i]
else:
    submission[LIST_CLASSES] = bag_by_average(test_predicts_list)
submission.to_csv(fp + "bag_by_mean.csv", index=False)

submission = new_submission.copy()
if add_comments:
    res = bag_by_average(test_predicts_list)
    for i,label in enumerate(LIST_CLASSES):

        submission[label] = res[:,i]
else:
    submission[LIST_CLASSES] = bag_by_rank_mean(test_predicts_list)
submission.to_csv(fp + "bag_by_rank_mean.csv", index=False)

submission = new_submission.copy()
submission[LIST_CLASSES] = bag_by_geomean(test_predicts_list)
submission.to_csv(fp + "bag_by_geomean.csv", index=False)