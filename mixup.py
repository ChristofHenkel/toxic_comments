"""
Exploration of mixup
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730
https://arxiv.org/abs/1710.09412

"""

import numpy as np
import tqdm
import random
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import wordnet as wn

def mixup_old( X, Y,alpha, portion):
    size = int(len(X) * portion)
    lam = np.random.beta(alpha, alpha, size)
    lambdas = lam.reshape(size, 1)

    indices = [ind for ind, x in enumerate(Y)]
    indices1 = np.random.permutation(indices)
    indices2 = np.random.permutation(indices1)


    x1, x2 = X[indices1][:size], X[indices2][:size]
    X_mixed = x1 * lambdas + x2 * (1 - lambdas)
    y1, y2 = Y[indices1][:size], Y[indices2][:size]
    Y_mixed = y1 * lambdas + y2 * (1 - lambdas)


    X_new = np.concatenate((X, X_mixed))
    Y_new = np.concatenate((Y, Y_mixed))
    old_indices = [ind for ind, x in enumerate(Y_new)]
    indices3 = np.random.permutation(old_indices)
    return X_new[indices3], Y_new[indices3]

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
    #old_indices = [ind for ind, x in enumerate(Y_new)]
    #indices3 = np.random.permutation(old_indices)
    # return X_new[indices3], Y_new[indices3]
    return  X_new, Y_new


def augment_with_translation_adhoc(list_of_sentences, portion):

    def translate_translate(comment, language):
        if hasattr(comment, "decode"):
            comment = comment.decode("utf-8")

        text = TextBlob(comment)
        try:
            text = text.translate(to=language)
            text = text.translate(to="en")
        except NotTranslated:
            pass

        return str(text)

    end = int(len(list_of_sentences) * portion)
    sentences = random.choices(list_of_sentences, k=end)
    new_sentences = []
    for sentence in tqdm.tqdm(sentences):
        lang = random.choice(['de','en','fr'])
        new_sentence = translate_translate(sentence,lang)
        new_sentences.append(new_sentence)
    list_of_sentences.extend(new_sentences)
    return random.shuffle(list_of_sentences)

def augmented_with_translation(train_data, portion, seed = 43, shuffle_result = True):

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