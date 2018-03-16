import pandas as pd
import numpy as np
import os
from global_variables import TRAIN_FILENAME, TEST_FILENAME, LIST_CLASSES, LIST_LOGITS

models = [fn.split('_predictions_train_oof.csv')[0] for fn in os.listdir('models/JAKUB/') if fn.endswith('_predictions_train_oof.csv')]

for m in models:
    os.mkdir('models/JAKUB/' + m + '/')

for m in models:
    j1 = pd.read_csv('models/JAKUB/'+ m + '_predictions_train_oof.csv', index_col=0)
    j1.drop(columns=['fold_id'], inplace=True)
    j1.columns = LIST_LOGITS

    df = pd.read_csv(TRAIN_FILENAME, index_col=0)
    df.drop(columns=['comment_text'], inplace=True)
    df2 = df.join(j1)
    df2.to_csv('models/JAKUB/'+ m + '/l2_train_data.csv')

from shutil import copyfile

for m in models:
    copyfile('models/JAKUB/'+ m + '_predictions_test_oof.csv','models/JAKUB/'+ m + '/l2_test_data.csv')


models = ['JAKUB/glove_lstm/',
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

csvs_test = ['models/' + m + 'l2_test_data.csv' for m in models]

for f in csvs_test:
    df = pd.read_csv(f)
    a = df.groupby('id').mean().reset_index()
    a.drop(columns = ['fold_id'], inplace = True)
    a.to_csv(f,index=False)