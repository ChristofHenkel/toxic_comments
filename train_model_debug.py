import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf
from keras.layers import CuDNNGRU, Dropout, Bidirectional, BatchNormalization, SpatialDropout1D
from tensorflow.contrib import layers
from spellchecker import Spellchecker
from collections import Counter
from utilities import get_oov_vector
import nltk
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors, FastText
import tqdm
import os
import time
from preprocess_utils import Preprocessor, preprocess
from augmentation import retranslation, mixup, synonyms
import pickle
from utilities import loadGloveModel, coverage
from global_variables import UNKNOWN_WORD, END_WORD, NAN_WORD, COMMENT, TRAIN_FILENAME, LIST_CLASSES, VALID_SLIM_FILENAME, TRAIN_SLIM_FILENAME, TEST_FILENAME

results = pd.DataFrame(columns=['fold_id','epoch','roc_auc_v','roc_auc_t','cost_val'])


class Config:

    train_fn = TRAIN_FILENAME
    do_preprocess = True
    add_polarity = False
    do_augmentation_with_translate = False
    do_augmentation_with_mixup = False
    if do_augmentation_with_mixup:
        mix_portion = 0.1
        alpha = 0.5
    do_synthezize_embeddings = False
    tokenize_mode = 'twitter'
    do_spellcheck_oov_words = False
    mode_embeddings = 'fasttext_300d'
    if do_synthezize_embeddings:
        synth_threshold = 0.7
    char_embedding_size = 256
    min_count_chars = 100
    bsize = 64
    max_seq_len = 300
    max_seq_len_chars = 500
    max_words = 200000
    rnn_units = 64
    att_size = 10
    fc_units = [256]
    epochs = 30
    model_name = 'gru_ATT_2'
    root = ''
    fp = 'models/RNN/' + model_name + '/'
    logs_path = fp + 'logs/'
    if not os.path.exists(root + fp):
        os.mkdir(root + fp)
    max_models_to_keep = 1
    save_by_roc = False
    level = ['word']
    lr = 0.00005
    decay = 1
    decay_steps = 400
    keep_prob = 0.5
    use_saved_embedding_matrix = True
    regularization_scale = None #0.0001
    char_vocab_size = 0

class ToxicComments:

    def __init__(self,cfg):
        self.preprocessor = Preprocessor()
        self.cfg = cfg
        self.word_counter = Counter()
        self.word2id = {}

    def tokenize_sentences(self,sentences, mode = 'twitter'):
        twitter_tokenizer = TweetTokenizer()
        tokenized_sentences = []
        print('tokenizing sentences using %s' %mode)
        for sentence in tqdm.tqdm(sentences,mininterval=5):
            if hasattr(sentence, "decode"):
                sentence = sentence.decode("utf-8")
            sentence = self.preprocessor.expand_contractions(sentence)
            if mode == 'nltk':
                tokens = nltk.tokenize.word_tokenize(sentence)
            elif mode == 'twitter':
                tokens = twitter_tokenizer.tokenize(sentence)
            else:
                tokens = None
            tokenized_sentences.append(tokens)
        return tokenized_sentences

    def create_word2id(self, list_of_tokenized_sentences):
        print('CREATING VOCABULARY')
        for tokenized_sentences in list_of_tokenized_sentences:
            for tokens in tqdm.tqdm(tokenized_sentences):
                self.word_counter.update(tokens)

        raw_counts = self.word_counter.most_common(self.cfg.max_words)
        vocab = [char_tuple[0] for char_tuple in raw_counts]
        print('%s words detected, keeping %s words' % (len(self.word_counter), len(vocab)))
        self.word2id = {word: (ind + 1) for ind, word in enumerate(vocab)}
        self.word2id[UNKNOWN_WORD] = len(self.word2id)
        self.id2word = dict((id, word) for word, id in self.word2id.items())
        print('finished')

    def tokenized_sentences2seq(self,tokenized_sentences, words_dict):
        print('converting to sequence')
        sequences = []
        for sentence in tqdm.tqdm(tokenized_sentences, mininterval=5):
            seq = []
            for token in sentence:
                try:
                    seq.append(words_dict[token])
                except KeyError:
                    seq.append(words_dict[UNKNOWN_WORD])
            sequences.append(seq)
        return sequences

    def update_words_dict(self,tokenized_sentences):
        self.word2id.pop(UNKNOWN_WORD, None)
        k = 0
        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in self.word2id:
                    k += 1
                    self.word2id[token] = len(self.word2id)
        print('{} words added'.format(k))
        self.word2id[UNKNOWN_WORD] = len(self.word2id)
        self.id2word = dict((id, word) for word, id in self.word2id.items())

    def clear_embedding_list(self,model, embedding_word_dict, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}
        k,l = 0, 0
        if self.cfg.do_spellcheck_oov_words:
            def P(word):
                return - embedding_word_dict.get(word, 0)

            def correction(word):
                return max(candidates(word), key=P)

            def candidates(word):
                return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

            def known(words):
                "The subset of `words` that appear in the dictionary of WORDS."
                return set(w for w in words if w in embedding_word_dict)

            def edits1(word):
                "All edits that are one edit away from `word`."
                letters = 'abcdefghijklmnopqrstuvwxyz'
                splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
                deletes = [L + R[1:] for L, R in splits if R]
                transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
                replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
                inserts = [L + c + R for L, R in splits for c in letters]
                return set(deletes + transposes + replaces + inserts)

            def edits2(word):
                "All edits that are two edits away from `word`."
                return (e2 for e1 in edits1(word) for e2 in edits1(e1))

        for word in tqdm.tqdm(words_dict):
            if word not in embedding_word_dict:
                l += 1
                if self.cfg.do_spellcheck_oov_words:
                    corrected_word = correction(word)
                    if corrected_word in embedding_word_dict:
                        row = model[corrected_word]
                        cleared_embedding_list.append(row)
                        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
                if self.cfg.do_synthezize_embeddings:

                    row = get_oov_vector(word, model, threshold=self.cfg.synth_threshold)
                    if row is None:
                        k += 1
                        continue
                    else:
                        cleared_embedding_list.append(row)
                        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
            else:
                row = model[word]
                cleared_embedding_list.append(row)
                cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
        print('embeddings not found: {0:.1f}%'.format(l / len(words_dict) * 100))
        print('embeddings not synthesized: {0:.1f}%'.format(k / len(words_dict) * 100))
        return cleared_embedding_list, cleared_embedding_word_dict

    def clear_embedding_list_fasttext(self,model, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}
        k = 0
        l = 0
        for word in tqdm.tqdm(words_dict):
            k+=1
            try:
                row = model[word]
                cleared_embedding_list.append(row)
                cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
            except KeyError:
                l +=1
                continue
        print('embeddings not found: {0:.1f}%'.format(l / len(words_dict) * 100))
        print('embeddings not synthesized: {0:.1f}%'.format(k / len(words_dict) * 100))
        return cleared_embedding_list, cleared_embedding_word_dict

    #def get_bad_sentences(self,vlosses, vlogits, X_valid, Y_valid):
    #    idx = (-vlosses).argsort()[:100]
    #    X = X_valid[idx]
    #    Y = Y_valid[idx]
    #    preds = np.concatenate((Y,vlogits[idx]))
    #    losses = vlosses[idx]
    #    sentences = []
    #    for row in X:
    #        sentences.append(' '.join([id_to_embedded_word[r] for r in row]))
    #    d = pd.DataFrame(preds, columns=list_classes.extend(['l' + label for label in list_classes]))
    #    #d[list_classes] = Y
    #    d['words'] = pd.Series(sentences)
    #    d['idx'] = pd.Series(idx)
    #    d['loss'] = pd.Series(losses)
    #    d.to_csv('misclassifies2.csv', index=False)

    def convert_tokens_to_ids(self,tokenized_sentences, embedding_word_dict):
        words_train = []
        'converting word index to embedding index'
        for sentence in tqdm.tqdm(tokenized_sentences):
            current_words = []
            for word_index in sentence:
                try:
                    word = self.id2word[word_index]
                    word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
                except KeyError:
                    word_id = embedding_word_dict.get(UNKNOWN_WORD, len(embedding_word_dict) - 2)
                current_words.append(word_id)

            if len(current_words) >= self.cfg.max_seq_len:
                current_words = current_words[:self.cfg.max_seq_len]
            else:
                current_words += [len(embedding_word_dict) - 1] * (self.cfg.max_seq_len - len(current_words))
            words_train.append(current_words)
        return words_train

    def prepare_embeddings(self, words_dict):
        print("Loading embeddings...")

        if self.cfg.mode_embeddings == 'fasttext_300d':
            print('loading Fasttext 300d')
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec', binary=False)
            embedding_word_dict = {w: ind for ind, w in enumerate(model.index2word)}
            embedding_size = 300
        elif self.cfg.mode_embeddings == 'mini_fasttext_300d':
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/mini_fasttext_300d2.vec',binary=False)
            embedding_word_dict = {w: ind for ind, w in enumerate(model.index2word)}
            embedding_size = 300
        elif self.cfg.mode_embeddings == 'fasttext_wiki_300d':
            model = FastText.load_fasttext_format('assets/embedding_models/ft_wiki/wiki.en.bin')
            embedding_word_dict = {w: ind for ind, w in enumerate(model.wv.index2word)}
            embedding_size = 300
        elif self.cfg.mode_embeddings == 'glove_300d':
            model = loadGloveModel('assets/embedding_models/glove/glove.840B.300d.txt',dims=300)
            embedding_word_dict = {w: ind for ind, w in enumerate(model)}
            embedding_size = 300
        elif self.cfg.mode_embeddings == 'glove_twitter_200d':
            model = loadGloveModel('assets/embedding_models/glove/glove.twitter.27B.200d.txt',dims=200)
            embedding_word_dict = {w: ind for ind, w in enumerate(model)}
            embedding_size = 200
        else:
            model = None
            embedding_size = None


        print("Preparing data...")
        if not self.cfg.mode_embeddings == 'fasttext_wiki_300d':
            embedding_list, embedding_word_dict = self.clear_embedding_list(model, embedding_word_dict, words_dict)
        else:
            embedding_list, embedding_word_dict = self.clear_embedding_list_fasttext(model, words_dict)

        del model

        embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
        embedding_list.append([0.] * embedding_size)
        embedding_word_dict[END_WORD] = len(embedding_word_dict)
        embedding_list.append([-1.] * embedding_size)

        embedding_matrix = np.array(embedding_list)


        id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())
        return embedding_matrix, embedding_word_dict, id_to_embedded_word

    def tokenize_list_of_sentences(self,list_of_sentences):

        list_of_tokenized_sentences = []
        for sentences in list_of_sentences:
            tokenized_sentences = self.tokenize_sentences(sentences, mode=self.cfg.tokenize_mode)

            # more preprocess on word level
            tokenized_sentences = [self.preprocessor.rm_hyperlinks(s) for s in tokenized_sentences]
            tokenized_sentences = [self.preprocessor.strip_spaces(s) for s in tokenized_sentences]
            list_of_tokenized_sentences.append(tokenized_sentences)

        return list_of_tokenized_sentences

    def save(self):
        with open(self.cfg.fp + 'tc.p','wb') as f:
            pickle.dump(self,f)

fold_count = 1
cfg = Config()

train_data = pd.read_csv(cfg.train_fn)
# t, v = train_test_split(train_data,test_size=0.2, random_state=123)
# t.to_csv("assets/raw_data/bagging_train.csv")
# v.to_csv("assets/raw_data/bagging_valid.csv")

#valid_data = pd.read_csv(VALID_SLIM_FILENAME)
test_data = pd.read_csv(TEST_FILENAME)
tc = ToxicComments(cfg)

if tc.cfg.do_preprocess:
    if tc.cfg.add_polarity:
        train_data = preprocess(train_data,add_polarity=True)
        #valid_data = preprocess(valid_data,add_polarity=True)
        test_data = preprocess(test_data, add_polarity=True)
    else:
        train_data = preprocess(train_data)
        #valid_data = preprocess(valid_data)
        test_data = preprocess(test_data)

sentences_train = train_data["comment_text"].fillna("_NAN_").values
#sentences_valid = valid_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values
Y = train_data[LIST_CLASSES].values

if 'word' in tc.cfg.level:
    #tokenized_sentences_train, tokenized_sentences_valid,tokenized_sentences_test = tc.tokenize_list_of_sentences([sentences_train,sentences_valid,sentences_test])
    tokenized_sentences_train,tokenized_sentences_test = tc.tokenize_list_of_sentences([sentences_train,sentences_test])


    #tc.create_word2id([tokenized_sentences_train,tokenized_sentences_valid,tokenized_sentences_test])
    tc.create_word2id([tokenized_sentences_train, tokenized_sentences_test])
    with open(tc.cfg.fp + 'tc_words_dict.p','wb') as f:
        pickle.dump(tc.word2id, f)

    sequences_train = tc.tokenized_sentences2seq(tokenized_sentences_train, tc.word2id)
    #sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, tc.words_dict)
    if cfg.use_saved_embedding_matrix:
        with open(tc.cfg.fp + 'embedding_word_dict.p','rb') as f:
            embedding_word_dict = pickle.load(f)
        embedding_matrix = np.load(tc.cfg.fp + 'embedding.npy')
        id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())

    else:
        embedding_matrix, embedding_word_dict, id_to_embedded_word = tc.prepare_embeddings(tc.word2id)
        coverage(tokenized_sentences_train,embedding_word_dict)
        with open(tc.cfg.fp + 'embedding_word_dict.p','wb') as f:
            pickle.dump(embedding_word_dict,f)
        np.save(tc.cfg.fp + 'embedding.npy',embedding_matrix)

    train_list_of_token_ids = tc.convert_tokens_to_ids(sequences_train, embedding_word_dict)
    #test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)

    X = np.array(train_list_of_token_ids)
    #X_test = np.array(test_list_of_token_ids)
    X_test = None
if 'char' in tc.cfg.level:
    tc.preprocessor.min_count_chars = tc.cfg.min_count_chars

    tc.preprocessor.create_char_vocabulary(sentences_train)
    with open(tc.cfg.fp + 'char2index.p','wb') as f:
        pickle.dump(tc.preprocessor.char2index,f)

    if 'word' in tc.cfg.level:
        Z = tc.preprocessor.char2seq(sentences_train, maxlen=tc.cfg.max_seq_len_chars)
        Z_test = None
        X = np.concatenate([X,Z], axis = 1)
        cfg.char_vocab_size = tc.preprocessor.char_vocab_size
    else:
        X = tc.preprocessor.char2seq(sentences_train, maxlen=tc.cfg.max_seq_len_chars)
        embedding_matrix = np.zeros((tc.preprocessor.char_vocab_size, tc.cfg.char_embedding_size))

        X_test = None


fold_size = len(X) // 10
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    if cfg.do_augmentation_with_mixup:
        X_train, Y_train = mixup( X_train, Y_train,cfg.alpha, cfg.mix_portion, seed=43)


def prelu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def squash(vector,epsilon=1e-9):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''

    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)

def routing(input, b_IJ, iter_routing=3, caps_dim_in=6, caps_dim_out=8, num_caps_out=6):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    bsize = input.get_shape()[0]
    num_caps_in = input.get_shape()[1]
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps_in, num_caps_out, caps_dim_in, caps_dim_out), dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, num_caps_out, 1, 1])
    W = tf.tile(W, [bsize, 1, 1, 1, 1])
    assert input.get_shape() == [bsize, num_caps_in, num_caps_out, caps_dim_in, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [bsize, num_caps_in, num_caps_out, caps_dim_out, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [bsize, 1, num_caps_out, caps_dim_out, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [bsize, 1, num_caps_out, caps_dim_out, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, num_caps_in, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [bsize, num_caps_in, num_caps_out, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return (v_J)

def spatial_dropout(x, keep_prob, seed=1234):
    # x is a convnet activation with shape BxWxHxF where F is the
    # number of feature maps for that layer
    # keep_prob is the proportion of feature maps we want to keep

    # get the batch size and number of feature maps
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[2]]

    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=x.dtype)

    # if we take the floor of this, we get a binary matrix where
    # (1-keep_prob)% of the values are 0 and the rest are 1
    binary_tensor = tf.floor(random_tensor)

    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, tf.shape(x)[2]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret

def _attention_mechanism(outputs, attention_layer_size, rnn_units,maxSeqLength):
        """
        Attention Network to average the output of the bidirectional RNN network.
        Small neural network, which should learn the 'importance' of an individual word for classification.
        :param outputs: The outut of the Rnn network.
        :param attention_layer_size: Size of the hidden layer.
        :return: alpha-coefficients which produce an weighted average of the rnn_outputs.
        """
        # Reshape outputs to tensor of shape (batch_size * self.maxSeqLength, 2 * self.lstm_size)
        # So each output from one Rnn cell is fed into the connected layer once in a time.
        outputs = tf.reshape(outputs, [-1, 2 * rnn_units])
        hidden_layer = layers.fully_connected(outputs, attention_layer_size, activation_fn=tf.nn.relu)
        attention_logits = layers.fully_connected(hidden_layer, 1, activation_fn=None)

        #Reshape attention_logits, such that is of the shape (batch_size, self.maxSeqLength, 1).
        attention_logits = tf.reshape(attention_logits, [-1, maxSeqLength, 1])

        # Apply softmax to the maxSeqLength dimensions, i.e. the different outputs.
        # alphas has shape (batch_size, self.maxSeqLength, 1)
        alphas = tf.nn.softmax(attention_logits, dim=1)

        # TODO: Idea for nomalisation of the softmax.
        # unnormed_softmax = tf.exp(attention_logits)
        # softmax_norm = tf.reduce_sum(unnormed_softmax * batch_masking, axis=1)
        # alphas = unnormed_softmax / softmax_norm

        # Store alphas as class variable for visualization of the attentions.


        return alphas



graph = tf.Graph()
bsize = cfg.bsize

with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)

    x = tf.placeholder(tf.int32, shape=(cfg.bsize, cfg.max_seq_len), name="x")
    y = tf.placeholder(tf.float32, shape=(cfg.bsize, 6), name="y")
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
    #keep_prob = tf.placeholder(shape=(),dtype=tf.float32, name="keep_prob")

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]],
                                    dtype=tf.float32, initializer=tf.constant_initializer(embedding_matrix),
                                    trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    x2 = spatial_dropout(embedded_input, keep_prob+0.3)
    x2 = Bidirectional(CuDNNGRU(cfg.rnn_units, return_sequences=True))(x2)
    alphas = _attention_mechanism(x2, attention_layer_size=cfg.att_size, rnn_units=cfg.rnn_units,
                                       maxSeqLength=cfg.max_seq_len)
    encodings = tf.reduce_sum(alphas * x2, 1)
    outputs = tf.transpose(x2, [0, 2, 1])
    maxs = tf.reduce_max(outputs, axis=2)
    means = tf.reduce_mean(outputs, axis=2)
    last = outputs[:, :, -1]
    x3 = tf.concat([maxs, means, last, encodings], axis=1)

    for num_units in cfg.fc_units:
        x3 = layers.fully_connected(x3, num_units, activation_fn=tf.nn.relu)
    x3 = tf.nn.dropout(x3, keep_prob=keep_prob)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)

    with tf.variable_scope('loss'):
        loss = binary_crossentropy(y, logits)
        cost = tf.losses.log_loss(predictions=logits, labels=y)
        (_, auc_update_op) = tf.metrics.auc(predictions=logits, labels=y, curve='ROC')

    with tf.variable_scope('optim'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=cfg.lr).minimize(loss)


train_iters = len(X_train) - (cfg.bsize * 2)
steps = train_iters // cfg.bsize
valid_iters = len(X_valid) - (cfg.bsize *2)




with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU': 0})) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(cfg.epochs):
        tic = time.time()
        costs = []
        step = 0
        tf.local_variables_initializer().run(session=sess)
        while step * cfg.bsize < train_iters:
            batch_x = X_train[step * cfg.bsize:(step + 1) * cfg.bsize]
            batch_y = Y_train[step * cfg.bsize:(step + 1) * cfg.bsize]
            #batch_z = Z_train[step * self.cfg.bsize:(step + 1) * self.cfg.bsize]
            cost_ , _, roc_auc_train = sess.run([cost,optimizer,auc_update_op],
                                                feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:cfg.keep_prob})
            if step % 10 == 0:
                print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,cfg.epochs,step,steps,cost_))
            costs.append(cost_)
            step += 1

        vstep = 0
        vcosts = []
        vlosses = np.asarray([])
        tf.local_variables_initializer().run(session=sess)
        while vstep * cfg.bsize < valid_iters:
            batch_x_valid = X_valid[vstep * cfg.bsize:(vstep + 1) * cfg.bsize]
            batch_y_valid = Y_valid[vstep * cfg.bsize:(vstep + 1) * cfg.bsize]
            #batch_z_valid = Z_valid[vstep * self.cfg.bsize:(vstep + 1) * self.cfg.bsize]
            test_cost_, valid_loss, roc_auc_valid = sess.run([cost,loss,auc_update_op],
                                                            feed_dict={x: batch_x_valid,
                                                   y: batch_y_valid,
                                                   keep_prob: 1
                                                   })
            vstep += 1
            vcosts.append(test_cost_)
            vlosses = np.concatenate((vlosses,valid_loss))
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        toc = time.time()
        print('time needed %s' %(toc-tic))
        print('valid loss: %s' % avg_cost)
        print('roc auc test : {:.4}'.format(roc_auc_valid))
        print('roc auc train : {:.4}'.format(roc_auc_train))
        avg_train_cost = np.log(np.mean(np.exp(costs[:valid_iters])))
        print('train loss %s' %avg_train_cost )









