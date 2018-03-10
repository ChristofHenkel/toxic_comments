import pandas as pd
from sklearn.preprocessing import minmax_scale
from global_variables import LIST_LOGITS, LIST_CLASSES
import numpy as np
from utilities import logloss
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe

models = ['models/CNN/SLIM/INCEPTION/inception2_slim/']


csvs_train = ['models/RNN/gru_ATT_4/l2_train_data.csv',
              'models/RNN/gru_ATT_6_glove/l2_train_data.csv',
              'models/CNN/inception5b/l2_train_data.csv']

def do_hyperopt(csvs_train, max_evals=100):
    dfs = [pd.read_csv(csv) for csv in csvs_train]

    xs = [df[LIST_LOGITS].values for df in dfs]
    ys = [df[LIST_CLASSES].values for df in dfs]
    for i, _ in enumerate(csvs_train[1:]):
        assert np.array_equal(ys[0], ys[i])

    space = [hp.uniform('w' + str(l) ,0,1) for l in range(len(xs))]

    def objective(space_elements):
        ws = [s/sum(space_elements) for s in space_elements]
        preds = ws[0] * xs[0]
        for l, _ in enumerate(space_elements[1:]):
            preds+= ws[l+1] * xs[l+1]
        return logloss(ys[0], preds)


    best = fmin(objective, space, algo=tpe.suggest, max_evals=len(csvs_train)*max_evals, verbose=True)
    ws = [best['w' + str(l)] for l in range(len(csvs_train))]
    ws = [s / sum(ws) for s in ws]
    print(ws)
    return ws

