import matplotlib
matplotlib.use('TkAgg')
import os
import pandas as pd
from gensim.models.fasttext import FastText
import numpy as np
from gensim import utils
from six import string_types, iteritems
import tqdm
from thesaurus import Word

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
    fn1 = 'assets/badwords.txt'
    fn2 = 'assets/swearWords.txt'
    with open(fn1) as f:
        content1 = [l.strip().lower() for l in f.readlines()]
    list_bad_words = []
    syns = {}
    for line in content1:
        items = line.split(', ')
        if len(items) == 1:
            if not items[0] in list_bad_words:
                list_bad_words.append(items[0])
        elif len(items) == 2:
            syns[items[0]] = items[1]
            if not items[1] in list_bad_words:
                list_bad_words.append(items[1])
    with open(fn2) as f:
        swearwords = [l.strip().lower() for l in f.readlines()]
    list_bad_words.extend(swearwords)
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
            splitLine = line.split(' ')
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



def get_oov_vector(word, model,dim = 300,minn=3, maxn = 6, threshold = 0.8):
    def get_ngrams(word, minn=3, maxn=6):
        def _get_gram(text, n):
            return [text[i:i + n] for i in range(len(text) - n + 1)]

        ngrams = []
        for n in range(minn, min(len(word) - 1, maxn) + 1):
            ngrams.extend(_get_gram(word, n))
        return ngrams
    ngrams = get_ngrams(word,minn=minn, maxn = maxn)
    vec = np.zeros(dim, dtype=np.float32)
    k = 0
    for gram in ngrams:
        try:
            vec += model[gram]
            k += 1
        except:
            pass
    if k != 0:
        vec /= k
        if k / (len(ngrams)+1) > threshold:
            return vec
    else:
        return None


def save_mini_fasttext_format(model, fname, words_dict, binary=False):
    """
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

     `fname` is the file used to save the vectors in
     `fvocab` is an optional file used to save the vocabulary
     `binary` is an optional boolean indicating whether the data is to be saved
     in binary word2vec format (default: False)
     `total_vec` is an optional parameter to explicitly specify total no. of vectors
     (in case word vectors are appended with document vectors afterwards)

    """

    total_vec = len(model.vocab)
    vector_size = model.syn0.shape[1]
    print("storing %sx%s projection weights into %s", total_vec, vector_size, fname)
    assert (len(model.vocab), vector_size) == model.syn0.shape
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            if word in words_dict:
                row = model.syn0[vocab.index]
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

def coverage(tokenized_sentences, embedding_word_dict):
    k = 0
    l = 0
    for tokenized_sentence in tqdm.tqdm(tokenized_sentences):
        for token in tokenized_sentence:
            l += 1
            if token not in embedding_word_dict:
                k += 1
    print('embeddings not found: {0:.1f}%'.format(k / l * 100))


def get_synonyms(words_dict):
    word_syns = {}
    for w in tqdm.tqdm(words_dict):
        word = Word(w)
        try:
            syns = word.synonyms(relevance=3)
        except:
            syns = None
        if syns is not None:
            word_syns[w] = syns
    return word_syns

def write_syns():
    raw_counts = list(tc.word_counter.items())
    vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > 100]
    word_syns = get_synonyms(vocab)


def write_config(fp,Config):
    pass

def save_runs():
    pass

