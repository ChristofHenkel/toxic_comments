import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
from global_variables import LIST_CLASSES, LIST_LOGITS
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from utilities import corr_matrix
import numpy as np

csvs_train = ['models/CNN/inception2_slim/inception2_slim_baggin_logits_folded.csv',
              #'models/NBSVM/slim/nbsvm_prediction_valid.csv',
              #'models/RNN/pavel_attention_slim2/baggin_logits_folded.csv',
              'models/RNN/pavel_all_outs_slim/birnn_all_outs_slim_baggin_logits_folded.csv'
              ]


dfs = [pd.read_csv(csv) for csv in csvs_train]
xs = [df[LIST_LOGITS].values for df in dfs]
n_models = len(csvs_train)

print('Corr matrix')
print(corr_matrix(xs))
print(' ')


for df in dfs:
    print(roc_auc_score(y_true=df[LIST_CLASSES].values, y_score=df[LIST_LOGITS].values))

def lloss(y_true,y_pred):
    l = 0
    for i in range(6):
        l += log_loss(y_true=y_true[:,i],y_pred=y_pred[:,i])
        l /= 6
    return l

ys = [df[LIST_CLASSES].values for df in dfs]

for i,_ in enumerate(csvs_train[1:]):
    assert np.array_equal(ys[0],ys[i])


Y = ys[0]
X = np.concatenate([xs])
X = X.transpose([1, 0, 2])

fold_size = len(X) // 10
means = np.zeros(6)
xgbs = np.zeros(6)
ws = np.zeros(6)
for i in range(6):

    for fold_id in range(0, 9):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        X_valid = X[fold_start:fold_end]
        Y_valid = Y[fold_start:fold_end]
        X_train = np.concatenate([X[:fold_start], X[fold_end:]])
        Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

        clf = XGBRegressor(objective='reg:logistic',max_depth=2,n_estimators=100,learning_rate=0.1,subsample=0.8, min_child_weight=3)
        clf.fit(X_train[:,:,i],Y_train[:,i])
        preds = clf.predict(X_valid[:,:,i])

        """
        xgdmat = xgb.DMatrix(X_train[:, :, i], Y_train[:, i])
        xgdmat2 = xgb.DMatrix(X_valid[:, :, i])
        our_params = {'subsample': 0.6,
                      'objective': 'reg:logistic', 'max_depth': 2, }
        # Grid Search CV optimized settings

        cv_xgb = xgb.cv(params=our_params, dtrain=xgdmat, num_boost_round=3000, nfold=5,
                        metrics=['rmse'],  # Make sure you enter metrics inside a list or you may encounter issues!
                        early_stopping_rounds=100, )
        final_gb = xgb.train(our_params, xgdmat, num_boost_round=cv_xgb.shape[0])

        preds2 = final_gb.predict(xgdmat2)
        """

        print('--------------------------------')
        print(mean_squared_error(y_true=Y_train[:,i],y_pred=X_train[:,0,i]))
        print(mean_squared_error(y_true=Y_train[:,i],y_pred=X_train[:,1,i]))
        #print(mean_squared_error(y_true=Y_train[:,i],y_pred=X_train[:,2,i]))

        means[i] += mean_squared_error(y_true=Y_valid[:,i],y_pred=np.mean(X_valid[:,:,i],axis= 1))
        xgbs[i] +=mean_squared_error(y_true=Y_valid[:,i],y_pred=preds)
        #ws[i] += mean_squared_error(y_true=Y_valid[:,i], y_pred=preds2)

        #print(mean_squared_error(y_true=Y_train[:, i], y_pred=0.8 * X_train[:, 1, i] + 0.2 * X_train[:, 0, i]))
        print(mean_squared_error(y_true=Y_train[:,i],y_pred=np.mean(X_train[:,:,i],axis= 1)))
        print(mean_squared_error(y_true=Y_valid[:,i],y_pred=preds))
        #print(mean_squared_error(y_true=Y_valid[:, i], y_pred=preds2))





print(mean_squared_error(y_true=Y_valid[:,i],y_pred=a))









dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

params = {
        'colsample_bytree':0.6,
        'learning_rate':0.1,'gamma':0,
        'max_depth':5, 'min_child_weight':1,
        'nthread':4,'reg_lambda':0,'reg_alpha':0,
        'objective':'reg:linear', 'seed':407,
        'silent':1, 'subsample':0.8
}