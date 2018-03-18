from global_variables import TRAIN_FILENAME, LIST_LOGITS, LIST_CLASSES
from hyperoptim import blend_with_hyperopt_weights, do_hyperopt_by_class
import hyperoptim
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from utilities import logloss, corr_matrix

models_own = ['NBSVM/word_model1/',
              'NBSVM/char_model1/',
              'RNN/gru_ATT_4/',
              'RNN/gru_ATT_6_glove/',
              'RNN/gru_CNN/',
              'XGB/xgb1/',
              'LGB/hurford2/',
              'CRNN/cudnn_rnn3/',
              'CAPS/cudrnn_caps/',
              'CNN/inception5b/',
              'RNN/HAN_2/',
              'RNN/HLSTM/',
                #'CAPS/CAPSNET/',
              'CAPS/CAPS3/',
                #'CAPS/CAPS2/',
              'CAPS/CAPS4/',
          'PUBLIC/RIDGE/',
          'CNN/DPCNN/',
          'CNN/DPCNN_3216/'

          ]

models_hans = ['HANS/gru_fix11/',
          'HANS/mengye_dpcnn/',
          'HANS/text_rcnn/'
    ]

models_jakub = ['JAKUB/glove_lstm/',
          'JAKUB/char_vdcnn/',
          'JAKUB/glove_dpcnn/',
          'JAKUB/fasttext_dpcnn/',
          'JAKUB/count_logreg/',
          'JAKUB/word2vec_lstm/',
          'JAKUB/glove_scnn/',
          'JAKUB/word2vec_gru/',
          'JAKUB/word2vec_scnn/',
          'JAKUB/fasttext_gru/',
          'JAKUB/tfidf_logreg/',
          'JAKUB/bad_word_logreg/',
          'JAKUB/fasttext_lstm/',
          'JAKUB/glove_gru/',
          'JAKUB/fasttext_scnn/',
          'JAKUB/word2vec_dpcnn/']

models = models_own + models_hans + models_jakub


csvs_train = ['models/' + m + 'l2_train_data.csv' for m in models]
csvs_test = ['models/' + m + 'l2_test_data.csv' for m in models]
dfs = [pd.read_csv(csv) for csv in csvs_train]
dfs_test = [pd.read_csv(csv) for csv in csvs_test]

def get_values(csv_files, columns, hstack = False, with_labels=True, scale = True):
    dfs = [pd.read_csv(csv) for csv in csv_files]
    if scale:
        for label in columns:
            for df in dfs:
                df[label] = np.clip(df[label].values,0,1)
                df[label] = minmax_scale(df[label])


    xs = [df[columns].values for df in dfs]
    if hstack:
        X = np.hstack(xs)
    else:
        X = np.concatenate([xs])
        X = X.transpose([1, 0, 2])

    if with_labels:
        ys = [df[LIST_CLASSES].values for df in dfs]

        for i, _ in enumerate(csv_files[1:]):
            assert np.array_equal(ys[0], ys[i])

        Y = ys[0]
        return X, Y
    else:
        return X



X, Y = get_values(csvs_train,columns=LIST_LOGITS,hstack=False,with_labels=True)

print('Corr matrix')
print(corr_matrix(list(X.transpose([1, 0, 2]))))
print(' ')

rocs = []
for m, model in enumerate(models):
    print('%s roc %s logloss %s'%(model,roc_auc_score(Y,X[:,m,:]),logloss(Y,X[:,m,:])))
    rocs.append(roc_auc_score(Y,X[:,m,:]))

kf = KFold(n_splits=10)
folder = kf.split(dfs[0])
#blends_valid = []
#blends_test = []
train_index, valid_index = folder.__next__()
dfs_train = [df.iloc[train_index] for df in dfs]
dfs_valid = [df.iloc[valid_index] for df in dfs]

rocs = []
for d, df in enumerate(dfs_train):
    print('%s roc %s logloss %s'%(d,roc_auc_score(df[LIST_CLASSES],df[LIST_LOGITS]),logloss(df[LIST_CLASSES].values,df[LIST_LOGITS].values)))
    rocs.append(roc_auc_score(df[LIST_CLASSES],df[LIST_LOGITS]))



start_id = int(np.argmax(rocs))
start_df = dfs_train[start_id]
start_model = models[start_id]
print('blend start is %s' % start_model)
blend_train = start_df
blend_valid = dfs_valid[start_id]
dfs_train = dfs_train[:start_id] + dfs_train[start_id+1:]
dfs_valid = dfs_valid[:start_id] + dfs_valid[start_id+1:]
rest_models = models[:start_id] + models[start_id+1:]

def find_next_df(blend,dfs_train):
    roc_scores = []
    for m, df in enumerate(dfs_train):

        dfs = [blend] + [df]

        weights = do_hyperopt_by_class(dfs)

        blend_tmp = blend_with_hyperopt_weights(dfs, weights)
        #blend_valid = blend_with_hyperopt_weights(dfs_valid, weights)

        roc_score = roc_auc_score(blend_tmp[LIST_CLASSES], blend_tmp[LIST_LOGITS])
        print(roc_score)

        roc_scores.append(roc_score)
        #print(roc_auc_score(blend_valid[LIST_CLASSES], blend_valid[LIST_LOGITS]))
    old_roc_score = roc_auc_score(blend[LIST_CLASSES], blend[LIST_LOGITS])

    next_df_id = int(np.argmax(roc_scores))
    if old_roc_score > max(roc_scores):
        print('no better blend found')
        next_df_id= None
    return next_df_id

def blend_with_next_df(blend,next_df):
    dfs = [blend] + [next_df]
    weights = do_hyperopt_by_class(dfs)
    blend = blend_with_hyperopt_weights(dfs, weights)
    return blend


for k in range(36):
    print('Iteration %s' % k)
    next_df_id = find_next_df(blend_train,dfs_train)
    if next_df_id is None:
        print('no better blend found')
        break
    print('blending with %s' % rest_models[next_df_id])
    blend_train= blend_with_next_df(blend_train,dfs_train[next_df_id])
    print('new roc train: %s' % roc_auc_score(blend_train[LIST_CLASSES], blend_train[LIST_LOGITS]))
    blend_valid = blend_with_next_df(blend_valid, dfs_valid[next_df_id])
    print('new roc valid: %s' % roc_auc_score(blend_valid[LIST_CLASSES], blend_valid[LIST_LOGITS]))
    dfs_train = dfs_train[:next_df_id] + dfs_train[next_df_id+1:]
    dfs_valid = dfs_valid[:next_df_id] + dfs_valid[next_df_id + 1:]
    rest_models = rest_models[:next_df_id] + rest_models[next_df_id + 1:]

