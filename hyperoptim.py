import pandas as pd
from sklearn.preprocessing import minmax_scale
from global_variables import LIST_LOGITS, LIST_CLASSES
import numpy as np
from utilities import logloss
from sklearn.metrics import roc_auc_score, log_loss
from hyperopt import hp, fmin, tpe
from global_variables import SAMPLE_SUBMISSION_FILENAME
from sklearn.model_selection import KFold

def do_hyperopt(csvs_train, max_evals=100):
    dfs = [pd.read_csv(csv) for csv in csvs_train]

    xs = [df[LIST_LOGITS].values for df in dfs]
    ys = [df[LIST_CLASSES].values for df in dfs]
    for i, _ in enumerate(csvs_train[1:]):
        assert np.array_equal(ys[0], ys[i])

    space = [hp.uniform('w' + str(l) ,0,1) for l in list(range(len(xs)))]

    def objective(space_elements):
        ws = [s/sum(space_elements) for s in space_elements]
        preds = ws[0] * xs[0]
        for l, _ in enumerate(space_elements[1:]):
            preds+= ws[l+1] * xs[l+1]
        return logloss(ys[0], preds)


    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, verbose=True)
    ws = [best['w' + str(l)] for l in range(len(csvs_train))]
    ws = [s / sum(ws) for s in ws]
    print(ws)
    return ws

def do_hyperopt_by_class(dfs, max_evals=50, verbose = False):
    weights = {}


    xs = [df[LIST_LOGITS].values for df in dfs]
    ys = [df[LIST_CLASSES].values for df in dfs]
    for i, _ in enumerate(dfs[1:]):
        assert np.array_equal(ys[0], ys[i])

    for i, class_name in enumerate(LIST_CLASSES):

        if verbose:
            print(class_name)
        xsi = [x[:,i] for x in xs]
        space = [hp.uniform('w' + str(l) ,0,1) for l in list(range(len(xs)))]

        def objective(space_elements):
            ws = [s/sum(space_elements) for s in space_elements]
            preds = ws[0] * xsi[0]
            for l, _ in enumerate(space_elements[1:]):
                preds+= ws[l+1] * xsi[l+1]
            return log_loss(ys[0][:,i], preds)


        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, verbose=True)
        ws = [best['w' + str(l)] for l in range(len(dfs))]
        ws = [s / sum(ws) for s in ws]
        if verbose:
            print(ws)
        weights[class_name] = ws
    return weights


def blend_with_hyperopt_weights(dfs, weights, test=False):
    blend = pd.DataFrame()
    if not test:
        blend[LIST_CLASSES] = dfs[0][LIST_CLASSES]
        for i, class_name in enumerate(LIST_CLASSES):
            blend['logits_' + class_name] = weights[class_name][0] * dfs[0]['logits_' + class_name]
            for j, df in enumerate(dfs[1:]):
                blend['logits_' + class_name] += weights[class_name][j + 1] * df['logits_' + class_name]
    else:
        for i, class_name in enumerate(LIST_CLASSES):
            blend[class_name] = weights[class_name][0] * dfs[0][class_name]
            for j, df in enumerate(dfs[1:]):
                blend[class_name] += weights[class_name][j + 1] * df[class_name]
    return blend

if __name__ == '__main__':
    csvs_train = ['models/STACKS/s9/l3_train_data.csv',
                  'models/STACKS/s10_jakub/l3_train_data.csv']

    csvs_test = ['models/STACKS/s9/l3_test_data.csv',
                  'models/STACKS/s10_jakub/l3_test_data.csv']

    dfs = [pd.read_csv(csv) for csv in csvs_train]
    dfs_test = [pd.read_csv(csv) for csv in csvs_test]

    kf = KFold(n_splits=10)
    blends_valid = []
    blends_test = []
    for train_index, valid_index in kf.split(dfs[0]):
        dfs_train = [df.iloc[train_index] for df in dfs]
        dfs_valid = [df.iloc[valid_index] for df in dfs]


        weights = do_hyperopt_by_class(dfs_train)

        blend_train = blend_with_hyperopt_weights(dfs_train,weights)
        blend_valid = blend_with_hyperopt_weights(dfs_valid, weights)


        print(roc_auc_score(blend_train[LIST_CLASSES],blend_train[LIST_LOGITS]))
        print(roc_auc_score(blend_valid[LIST_CLASSES], blend_valid[LIST_LOGITS]))

        blend_test = blend_with_hyperopt_weights(dfs_test,weights,test=True)
        blends_valid.append(blend_valid)
        blends_test.append(blend_test)

    blend_valid = pd.concat(blends_valid)
    sub = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)
    sub[LIST_CLASSES] = blend_test[LIST_CLASSES]
    sub.to_csv('l4_test.csv',index=False)



