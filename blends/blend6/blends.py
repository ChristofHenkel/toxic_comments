# All credits goes to original authors.. Just another blend...
import pandas as pd
from sklearn.preprocessing import minmax_scale


# 98.60
#sup = pd.read_csv('../input/blend-of-blends-1/superblend_1.csv')
#allave = pd.read_csv('../input/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv')
#gru = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv')

#blend = allave.copy()
#col = blend.columns

#col = col.tolist()
#col.remove('id')
# keeping weight of single best model higher than other blends..
#blend[col] = 0.2*minmax_scale(allave[col].values)+0.6*minmax_scale(gru[col].values)+0.2*minmax_scale(sup[col].values)
#print('stay tight kaggler')
#blend.to_csv("hight_of_blend_v2.csv", index=False)




################################################
# blend  -->    98.64
sup = pd.read_csv('models/PUBLIC/blend-of-blends-1/superblend_1.csv')
allave = pd.read_csv('models/PUBLIC/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv')
own_stack = pd.read_csv('models/STACKS/s1/s1_stacked_xgb_nn.csv')

blend = allave.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.2*minmax_scale(allave[col].values)+0.6*minmax_scale(own_stack[col].values)+0.2*minmax_scale(sup[col].values)
print('stay tight kaggler')
blend.to_csv("blends/blend6/blend6.csv", index=False)


