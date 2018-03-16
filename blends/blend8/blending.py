import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

stack3_slim_train = pd.read_csv('models/STACKS/s3_SLIM_Inception_and_RNN/l4_train_data.csv')
stack4_train = pd.read_csv('models/STACKS/s4/l3_train_data.csv')

stack4_slim_train, v = train_test_split(stack4_train,test_size=0.2, random_state=123)






blend = allave.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.15*minmax_scale(allave[col].values)+0.7*minmax_scale(own_stack[col].values)+0.15*minmax_scale(sup[col].values)
print('stay tight kaggler')
blend.to_csv("blends/blend7/blend7.csv", index=False)


