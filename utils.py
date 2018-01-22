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

def load_bad_words():
    fn = 'assets/badwords.txt'
    with open(fn) as f:
        content = [l.strip().lower() for l in f.readlines()]
    list_bad_words = []
    syns = {}
    for line in content:
        items = line.split(', ')
        if len(items) == 1:
            if not items[0] in list_bad_words:
                list_bad_words.append(items[0])
        elif len(items) == 2:
            syns[items[0]] = items[1]
            if not items[1] in list_bad_words:
                list_bad_words.append(items[1])
    return list_bad_words, syns


def create_embedding_matrix(X, word2index, mode = 'fasttext'):

    fasttext_fn = 'assets/embedding_models/ft_reviews_dim100_w5_min50/ft_reviews_dim100_w5_min50.ft_model'
    glove_fn = 'assets/embedding_models/glove/glove.twitter.27B.100d.txt'

    if mode == 'fasttext':
        model = FastText.load(fasttext_fn)
        dims = model.layer1_size
    elif mode == 'glove':
        model = loadGloveModel(glove_fn)
        dims = 100
    else:
        model = None
        dims = 0
    index2word = {ind: w for w, ind in word2index.items()}
    a = np.ndarray.flatten(X)
    indices = list(set(a))

    matrix = np.zeros((max(indices)+1,dims),dtype=np.float32)
    j = 0
    for k, ind in enumerate(indices):
        try:
            word = index2word[ind]
            vec = model[word]
            j += 1
        except:
            vec = np.random.uniform(-1,1,dims).astype(np.float32)
        matrix[ind] = vec
    print(' %s perc in model' %(j / max(indices)))
    return matrix

def loadGloveModel(gloveFile, dims = 100):
    print("Loading Glove Model")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            if embedding.shape[0] == dims:
                model[word] = embedding
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
    return model

def load_glove_embedding(word_index,dims = 100,max_features=1000000):
    model = loadGloveModel('assets/embedding_models/glove/glove.twitter.27B.100d.txt',dims)
    emb_mean, emb_std = 0.02631, 0.58371
    if dims == 50: emb_mean, emb_std = 0.04399, 0.73192
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, dims))
    j = 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            j += 1
    print(' %s perc in model' % (j / nb_words))
    return embedding_matrix




def list_seq2word_dict(list_seq,word2index):
    index2word = {ind:w for w, ind in word2index.items()}
