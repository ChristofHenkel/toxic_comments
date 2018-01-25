"""
Exploration of mixup
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730
https://arxiv.org/abs/1710.09412

"""

import numpy as np
import tqdm
from random import choices, choice, shuffle
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

def mixup( X, Y,alpha, portion):
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


def augment_with_translation(list_of_sentences, portion):

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
    sentences = choices(list_of_sentences, k=end)
    new_sentences = []
    for sentence in tqdm.tqdm(sentences):
        lang = choice(['de','en','fr'])
        new_sentence = translate_translate(sentence,lang)
        new_sentences.append(new_sentence)
    list_of_sentences.extend(new_sentences)
    return shuffle(list_of_sentences)







