import pandas as pd
from sklearn.preprocessing import minmax_scale
from global_variables import LIST_LOGITS, LIST_CLASSES
import numpy as np
from utilities import logloss
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe

csvs_train = ['models/RNN/gru_ATT_4/l2_train_data.csv',
              'models/RNN/gru_ATT_6_glove/l2_train_data.csv',
              'models/CNN/inception5b/l2_train_data.csv']

def do_hyperopt(csvs_train, csvs_test, max_evals, level = 'classes'):
    dfs = [pd.read_csv(csv) for csv in csvs_train]

    xs = [df[LIST_LOGITS].values for df in dfs]
    ys = [df[LIST_CLASSES].values for df in dfs]
    for i, _ in enumerate(csvs_train[1:]):
        assert np.array_equal(ys[0], ys[i])

    def objective(w):
        preds = np.array(w[c]) * xs[c] + np.array(w[1]) * xs[1] + (1 - np.array(w[0]) - np.array(w[1])) * xs[2]
        return logloss(ys[0], preds)


dfs = [pd.read_csv(csv) for csv in csvs_train]

xs = [df[LIST_LOGITS].values for df in dfs]
ys = [df[LIST_CLASSES].values for df in dfs]
for i, _ in enumerate(csvs_train[1:]):
    assert np.array_equal(ys[0], ys[i])


# define an objective function
def objective(w):
    preds = np.array(w[0]) * xs[0] + np.array(w[1]) * xs[1] + (1-np.array(w[0])-np.array(w[1])) * xs[2]
    return logloss(ys[0],preds)

# define a search space

space = [[hp.uniform('w' + str(k) + '_'+ l, 0, 1) for l in LIST_CLASSES] for k in range(len(xs)-1)]

# minimize the objective over the space

best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, verbose = True)

print(best)

w = [[best['w' + str(k) + '_' + l] for l in LIST_CLASSES]for k in range(len(xs)-1)]
preds = np.array(w[0]) * xs[0] + np.array(w[1]) * xs[1] + (1-np.array(w[0])-np.array(w[1])) * xs[2]
for x in xs:
    print(logloss(ys[0],x))
print(logloss(ys[0],preds))
print(logloss(ys[0],np.mean(xs[:2],axis = 0)))
#logloss(ys[0],preds)

def objective(w):
    preds = np.array(w) * xs[0] + (1-np.array(w)) * xs[1]
    return logloss(ys[0],preds)

# define a search space

space = [hp.uniform('w_'+ l, 0, 1) for l in LIST_CLASSES]

# minimize the objective over the space

best = fmin(objective, space, algo=tpe.suggest, max_evals=500, verbose = True)

print(best)

w = [best['w_' + l] for l in LIST_CLASSES]
preds = np.array(w) * xs[0] + (1-np.array(w)) * xs[1]
for x in xs:
    print(logloss(ys[0],x))
print(logloss(ys[0],preds))

def objective(w):
    preds2 = np.array(w) * preds + (1-np.array(w)) * xs[2]
    return logloss(ys[0],preds2)

# define a search space

space = [hp.uniform('w_'+ l, 0, 1) for l in LIST_CLASSES]

# minimize the objective over the space

best = fmin(objective, space, algo=tpe.suggest, max_evals=500, verbose = True)

print(best)

w = [best['w_' + l] for l in LIST_CLASSES]
preds2 = np.array(w) * preds + (1-np.array(w)) * xs[2]
for x in xs:
    print(logloss(ys[0],x))
print(logloss(ys[0],preds2))
print(logloss(ys[0],np.mean(xs, axis = 0)))

print(roc_auc_score(ys[0],preds2))
print(roc_auc_score(ys[0],np.mean(xs, axis = 0)))