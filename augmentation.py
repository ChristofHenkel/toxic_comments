"""
Exploration of mixup
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730
https://arxiv.org/abs/1710.09412

"""

import numpy as np
import tqdm
import pandas as pd
from sklearn.utils import shuffle
import pickle
import random


def mixup( X_train, Y_train,alpha, portion, seed):
    np.random.seed(seed)
    size = int(X_train.shape[0] * portion)
    lambdas = np.random.beta(alpha, alpha, size)
    lambdas = lambdas.reshape(size, 1)

    indices = [ind for ind, x in enumerate(Y_train)]
    indices1 = np.random.permutation(indices)
    indices2 = np.random.permutation(indices1)

    # get unpadded x1, x2

    x1, x2 = X_train[indices1][:size], X_train[indices2][:size]
    pad = X_train[0][-1]
    X_mixed = np.ones((x1.shape), dtype=np.int32) * pad
    for k in tqdm.tqdm(range(size)):
        lam = lambdas[k][0]
        seq1 = [item for item in x1[k] if not item == pad]
        seq2 = [item for item in x2[k] if not item == pad]
        end1 = int(len(seq1) * lam)
        beg2 = int(len(seq2) * (1-lam))
        new_seq = seq1[:end1] + seq2[beg2:]
        X_mixed[k][:len(new_seq)] = new_seq[:X_train.shape[1]]

    y1, y2 = Y_train[indices1][:size], Y_train[indices2][:size]
    Y_mixed = y1 * lambdas + y2 * (1 - lambdas)


    X_new = np.concatenate((X_train, X_mixed))
    Y_new = np.concatenate((Y_train, Y_mixed))
    X_new, Y_new = shuffle(X_new,Y_new, random_state = 42)
    return  X_new, Y_new


def retranslation(train_data, portion, seed = 43, shuffle_result = True):

    train_data_fr = pd.read_csv("assets/raw_data/train_fr.csv")
    train_data_de = pd.read_csv("assets/raw_data/train_de.csv")
    train_data_es = pd.read_csv("assets/raw_data/train_es.csv")
    data = pd.concat([train_data_fr,train_data_de,train_data_es])

    ids = train_data['id'].values
    data = data[data['id'].isin(ids)]
    frac = (train_data.shape[0] * portion) / data.shape[0]
    data = data.sample(frac = frac, random_state=seed)
    data = pd.concat([train_data, data])
    if shuffle_result:
        data = data.sample(frac=1, random_state=seed)
    return data

def synonyms(tokenized_sentences,Y_train, portion):
    with open('word_syns.p', 'rb') as f:
        word_syns = pickle.load(f)

    new_sentences = []
    new_Y = []
    Y_train = list(Y_train)
    indices = random.sample(range(len(tokenized_sentences)),int(len(tokenized_sentences)*portion))
    s_to_check = [tokenized_sentences[k] for k in indices]
    y_to_check= [Y_train[k] for k in indices]
    for k,sentence in enumerate(s_to_check):
        new_sentence = [w for w in sentence]
        replaceable_indices = [ind for ind, w in enumerate(sentence) if w in word_syns]
        if replaceable_indices != []:
            num_words = np.random.geometric(0.5)
            num_words = min(num_words,len(replaceable_indices))
            inds = np.random.choice(replaceable_indices,size=num_words,replace=False)
            for i in inds:
                syns = word_syns[sentence[i]]
                syn_id = np.random.geometric(0.5)
                try:
                    new_word = syns[syn_id]
                except:
                    try:
                        new_word = syns[-1]
                    except:
                        new_word = []
                if new_word != []:
                    new_sentence[i] = new_word
            if len(inds) > 0:
                new_sentences.append(new_sentence)
                new_Y.append(y_to_check[k])
    tokenized_sentences.extend(new_sentences)
    Y_train.extend(new_Y)
    return tokenized_sentences, Y_train




#for synset in wn.synsets('cat'):
#    for lemma in synset.lemmas():
#        print(lemma.name())##
#
#a = wn.synsets('dog')
#wn.synsets('cat').lemma_names()
#
#from thesaurus import Word#
#
#w = Word('fuck')
#w.synonyms(relevance=[2,3])