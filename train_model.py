import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf
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
from architectures import CNN, CAPS, BIRNN, CRNN, DENSE, CNNRNN, HYBRID
import pickle
from utilities import loadGloveModel, coverage
from global_variables import UNKNOWN_WORD, END_WORD, NAN_WORD, COMMENT, TRAIN_FILENAME, LIST_CLASSES, VALID_SLIM_FILENAME, TRAIN_SLIM_FILENAME, TEST_FILENAME

model_baseline = BIRNN().gru_ATT_6

results = pd.DataFrame(columns=['fold_id','epoch','roc_auc_v','roc_auc_t','cost_val'])

# TODO add decay
#global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.1
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                       100000, 0.96, staircase=True)

## Passing global_step to minimize() will increment it at each step.
#learning_step = (
#tf.train.GradientDescentOptimizer(learning_rate)
#.minimize(...my loss..., global_step=global_step))


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
    bsize = 512
    max_seq_len = 300
    max_seq_len_chars = 500
    max_words = 200000
    rnn_units = 64
    att_size = 10
    fc_units = [256]
    epochs = 30
    model_name = 'gru_ATT_4'
    root = ''
    fp = 'models/RNN/' + model_name + '/'
    logs_path = fp + 'logs/'
    if not os.path.exists(root + fp):
        os.mkdir(root + fp)
    max_models_to_keep = 1
    save_by_roc = False
    level = ['word']
    lr = 0.0004
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


class Model:

    def __init__(self, cfg):
        self.cfg = cfg
        self.graph = tf.Graph()

    def write_config(self):
        with open(os.path.join(self.cfg.fp, 'config.txt'), 'w') as f:
            f.write('Baseline = {}\n'.format(model_baseline.__name__))
            f.write('\n')
            f.write('Config\n')
            for line in self.class2list(Config):
                f.write('{} = {}\n'.format(line[0], line[1]))

    @staticmethod
    def class2list(class_):
        class_list = [[item,class_.__dict__ [item]]for item in sorted(class_.__dict__ ) if not item.startswith('__')]
        return class_list

    def set_graph(self, embedding_matrix):
        with self.graph.as_default():
            # tf Graph input
            tf.set_random_seed(1)

            if 'word' and 'char' in self.cfg.level:
                self.x = tf.placeholder(tf.int32, shape=(self.cfg.bsize, self.cfg.max_seq_len + self.cfg.max_seq_len_chars), name="x")
            elif 'word' in self.cfg.level:
                self.x = tf.placeholder(tf.int32, shape=(self.cfg.bsize, self.cfg.max_seq_len), name="x")
            else:
                self.x = tf.placeholder(tf.int32, shape=(self.cfg.bsize, self.cfg.max_seq_len_chars), name="x")

            self.y = tf.placeholder(tf.float32, shape=(self.cfg.bsize,6), name="y")
            #self.em = tf.placeholder(tf.float32, shape=(embedding_matrix.shape[0], embedding_matrix.shape[1]), name="em")
            self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

            #self.output = model_baseline(self.em,self.x,self.keep_prob, self.cfg)
            self.output = model_baseline(embedding_matrix, self.x, self.keep_prob, self.cfg)


            with tf.variable_scope('logits'):
                self.logits = self.output

            with tf.variable_scope('loss'):
                self.loss = binary_crossentropy(self.y,self.logits)
                self.cost = tf.losses.log_loss(predictions=self.logits, labels=self.y)
                (_, self.auc_update_op) = tf.metrics.auc(predictions=self.logits,labels=self.y,curve='ROC')
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.cfg.lr, self.global_step,self.cfg.decay_steps, self.cfg.decay, staircase=True)

            with tf.variable_scope('optim'):
                #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step=self.global_step)


            with tf.variable_scope('saver'):
                self.saver = tf.train.Saver(max_to_keep=self.cfg.max_models_to_keep)

    def save(self, sess, fold_id, epoch,roc_auc_valid,roc_auc_train,cost_val):
        results.loc[len(results)] = [fold_id, epoch, roc_auc_valid, roc_auc_train,cost_val]
        results.to_csv(self.cfg.fp + 'results.csv')
        df = results.loc[results['fold_id'] == fold_id]

        if self.cfg.save_by_roc:
            do_save = roc_auc_valid >= df[['roc_auc_v']].apply(max, axis=0)[0]
        else:
            do_save = cost_val <= df[['cost_val']].apply(min, axis=0)[0]

        if do_save:
            print('saving model...', end='')
            model_name = 'k%s_e%s.ckpt' % (fold_id,epoch)
            s_path = self.saver.save(sess, self.cfg.logs_path + model_name)
            print("Model saved in file: %s" % s_path)




    def train(self, X_train, Y_train, X_valid, Y_valid, X_test, embedding_matrix, fold_id=0, do_submission = False):

        self.write_config()
        if do_submission:
            sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
        train_iters = len(X_train) - (self.cfg.bsize * 2)
        steps = train_iters // self.cfg.bsize
        valid_iters = len(X_valid) - (self.cfg.bsize *2)

        with tf.Session(graph=self.graph) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(self.cfg.epochs):
                tic = time.time()
                costs = []
                step = 0
                tf.local_variables_initializer().run(session=sess)
                while step * self.cfg.bsize < train_iters:
                    batch_x = X_train[step * self.cfg.bsize:(step + 1) * self.cfg.bsize]
                    batch_y = Y_train[step * self.cfg.bsize:(step + 1) * self.cfg.bsize]
                    #batch_z = Z_train[step * self.cfg.bsize:(step + 1) * self.cfg.bsize]
                    cost_ , _, roc_auc_train = sess.run([self.cost,self.optimizer,self.auc_update_op],
                                                        feed_dict={self.x:batch_x,
                                                                     self.y:batch_y,
                                                                   #self.em:embedding_matrix,
                                                                     self.keep_prob:self.cfg.keep_prob})
                    if step % 10 == 0:
                        print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,self.cfg.epochs,step,steps,cost_))
                    costs.append(cost_)
                    step += 1

                vstep = 0
                vcosts = []
                vlosses = np.asarray([])
                tf.local_variables_initializer().run(session=sess)
                while vstep * self.cfg.bsize < valid_iters:
                    batch_x_valid = X_valid[vstep * self.cfg.bsize:(vstep + 1) * self.cfg.bsize]
                    batch_y_valid = Y_valid[vstep * self.cfg.bsize:(vstep + 1) * self.cfg.bsize]
                    #batch_z_valid = Z_valid[vstep * self.cfg.bsize:(vstep + 1) * self.cfg.bsize]
                    test_cost_, valid_loss, roc_auc_valid, used_lr = sess.run([self.cost,self.loss,self.auc_update_op,self.learning_rate],
                                                                    feed_dict={self.x: batch_x_valid,
                                                           self.y: batch_y_valid,
                                                        #self.em: embedding_matrix,
                                                           self.keep_prob: 1
                                                           })
                    vstep += 1
                    vcosts.append(test_cost_)
                    vlosses = np.concatenate((vlosses,valid_loss))
                avg_cost = np.log(np.mean(np.exp(vcosts)))
                toc = time.time()
                print('time needed %s' %(toc-tic))
                print('learning_rate %s' % used_lr)
                print('valid loss: %s' % avg_cost)
                print('roc auc test : {:.4}'.format(roc_auc_valid))
                print('roc auc train : {:.4}'.format(roc_auc_train))
                avg_train_cost = np.log(np.mean(np.exp(costs[:valid_iters])))
                print('train loss %s' %avg_train_cost )

                self.save(sess,fold_id,epoch,roc_auc_valid,roc_auc_train,avg_cost)
                if do_submission:
                    self.populate_submission(X_test,sess,epoch, roc_auc_valid,roc_auc_train, sample_submission,fold_id)


    def populate_submission(self,X_test,sess, epoch, roc_auc_test,roc_auc_train, sample_submission,fold_id):

        num_batches = (len(X_test) // self.cfg.bsize) + 1
        res = np.zeros((len(X_test), 6))
        for s in range(num_batches):
            if s % 50 == 0:
                print(s)
            batch_x_test = X_test[s * self.cfg.bsize:(s + 1) * self.cfg.bsize]

            logits_ = sess.run(self.logits, feed_dict={self.x: batch_x_test,
                                                  self.keep_prob: 1})

            res[s * self.cfg.bsize:(s + 1) * self.cfg.bsize] = logits_

        sample_submission[LIST_CLASSES] = res

        dir_name = self.cfg.model_name
        if not os.path.exists('submissions/' + dir_name):
            os.mkdir('submissions/' + dir_name)
        fn = "submissions/"
        fn += dir_name + "model_k"+ str(fold_id)
        fn += 'e' + str(epoch)

        fn += "v"+ str(round(roc_auc_test,ndigits=4))
        fn += "t"+ str(round(roc_auc_train,ndigits=4)) + ".csv"
        sample_submission.to_csv(fn, index=False)




def train_folds(fold_count=10):

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

        m = Model(cfg)
        m.set_graph(embedding_matrix)
        m.train(X_train, Y_train, X_valid, Y_valid, X_test, embedding_matrix, fold_id)


#if __name__ == '__main__':
train_folds(fold_count=10)
