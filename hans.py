import pandas as pd
from global_variables import TRAIN_FILENAME, LIST_CLASSES, LIST_LOGITS

own = pd.read_csv('models/CNN/DPCNN/l2_test_data.csv')
train = pd.read_csv(TRAIN_FILENAME, index_col=0)
gru_fix11 = pd.read_csv('models/HANS/gru_fix11_stack_train_oof.csv', index_col=6)
gru_fix11.columns = LIST_LOGITS
t = train.join(gru_fix11)
t.drop(columns = ['comment_text'], inplace=True)
t.to_csv('models/HANS/gru_fix11/l2_train_data.csv')

mengye = pd.read_csv('models/HANS/mengye_dpcnn_stack_train_oof.csv', index_col=6)
mengye.columns = LIST_LOGITS
t = train.join(mengye)
t.drop(columns = ['comment_text'], inplace=True)
t.to_csv('models/HANS/mengye_dpcnn/l2_train_data.csv')

text_rcnn = pd.read_csv('models/HANS/text_rcnn_att_stack_train_oof.csv', index_col=6)
text_rcnn.columns = LIST_LOGITS
t = train.join(text_rcnn)
t.drop(columns = ['comment_text'], inplace=True)
t.to_csv('models/HANS/text_rcnn/l2_train_data.csv')


gru_fix11_test = pd.read_csv('models/HANS/gru_fix11_stack_test.csv')