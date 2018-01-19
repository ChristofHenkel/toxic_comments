import matplotlib
matplotlib.use('TkAgg')
import os
import pandas as pd
from gensim.models.fasttext import FastText
import numpy as np


labels = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']
label2id = {name:id for id,name in enumerate(labels)}
id2label = {id:name for id,name in enumerate(labels)}
raw_data_dir = 'assets/raw_data/'
train_fn = 'train.csv'
test_fn = 'test.csv'

def read_metadata(data_dir, filename):
    meta_filepath = os.path.join(data_dir, filename)
    meta_data = pd.read_csv(meta_filepath)
    return meta_data

def read_data(data_dir, filename, mode = 'train'):
    meta_data = read_metadata(data_dir, filename)
    data = meta_data.to_dict('index')
    data = list(data.values())
    if mode == 'train':
        data = [{'id': item['id'], 'text': item['comment_text'], 'label': [item[label] for label in labels]} for item in data]
    else:
        data = [{'id': item['id'], 'text': item['comment_text'], 'label': []} for item in data]
    return data

def create_textcorpus(corpus,fp_out):
    textcorpus = [item['text'] for item in corpus]
    with open(fp_out,'w') as f:
        f.writelines(textcorpus)


def create_embedding_matrix(X, word2index):
    model = FastText.load('assets/embedding_models/ft_reviews_dim100_w5_min50/ft_reviews_dim100_w5_min50.ft_model')
    index2word = {ind: w for w, ind in word2index.items()}
    a = np.ndarray.flatten(X)
    indices = list(set(a))

    matrix = np.zeros((max(indices)+1,model.layer1_size),dtype=np.float32)
    for ind in indices:
        try:
            word = index2word[ind]
            vec = model[word]
        except:
            print(str(ind) + ' error')
            vec = np.random.uniform(-1,1,model.layer1_size).astype(np.float32)
        matrix[ind] = vec
    return matrix

def list_seq2word_dict(list_seq,word2index):
    index2word = {ind:w for w, ind in word2index.items()}
