import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf

from utilities import get_oov_vector
import nltk
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
import tqdm
import os
import time
from preprocess_utils import Preprocessor
from mixup import augmented_with_translation, mixup
from architectures import pavel2 as model_baseline


class Config:

    unknown_word = "_UNK_"
    end_word = "_END_"
    nan_word = "_NAN_"
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class PreprocessConfig:

    max_sentence_len = 500
    do_augmentation_with_translate = False
    do_augmentation_with_mixup = False
    do_synthezize_embeddings = False


class ModelConfig:
    bsize = 512
    max_seq_len = 500
    epochs = 12
    model_name = 'pavel36/'


class ToxicComments:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.cfg = Config()
        self.pcfg = PreprocessConfig()

        self.train_data = pd.read_csv("assets/raw_data/train.csv")
        self.test_data = pd.read_csv("assets/raw_data/test.csv")

        self.sentences_train = self.train_data["comment_text"].fillna("_NAN_").values
        self.sentences_test = self.test_data["comment_text"].fillna("_NAN_").values

    def get_augmented_data(self):
        if self.pcfg.do_augmentation_with_translate:
            train_data = augmented_with_translation(self.train_data, 0.3)
        if self.pcfg.do_augmentation_with_mixup:
            train_data = mixup()

    def tokenize_sentences(self,sentences, words_dict, mode = 'nltk'):
        twitter_tokenizer = TweetTokenizer()
        tokenized_sentences = []
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
            result = []
            for word in tokens:
                word = word.lower()
                if word not in words_dict:
                    words_dict[word] = len(words_dict)
                word_index = words_dict[word]
                result.append(word_index)
            tokenized_sentences.append(result)
        return tokenized_sentences, words_dict

    def clear_embedding_list(self,model, embedding_word_dict, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}
        k = 0
        l = 0
        for word in tqdm.tqdm(words_dict):
            if word not in embedding_word_dict:
                if self.pcfg.do_synthezize_embeddings:
                    l += 1
                    row = get_oov_vector(word, model, threshold=0.7)
                    if row is None:
                        k += 1
                        continue
                    else:
                        cleared_embedding_list.append(row)
                        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
                else:
                    continue
            else:
                row = model[word]
                cleared_embedding_list.append(row)
                cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
        print('embeddings not found: {0:.1f}%'.format(l / len(words_dict) * 100))
        print('embeddings not synthesized: {0:.1f}%'.format(k / len(words_dict) * 100))
        return cleared_embedding_list, cleared_embedding_word_dict

    def get_bad_sentences(self,vlosses, vlogits, X_valid, Y_valid):
        idx = (-vlosses).argsort()[:100]
        X = X_valid[idx]
        Y = Y_valid[idx]
        preds = np.concatenate((Y,vlogits[idx]))
        losses = vlosses[idx]
        sentences = []
        for row in X:
            sentences.append(' '.join([id_to_embedded_word[r] for r in row]))
        d = pd.DataFrame(preds, columns=self.cfg.list_classes.extend(['l' + label for label in cfg.list_classes]))
        #d[list_classes] = Y
        d['words'] = pd.Series(sentences)
        d['idx'] = pd.Series(idx)
        d['loss'] = pd.Series(losses)
        d.to_csv('misclassifies2.csv', index=False)

    def convert_tokens_to_ids(self,tokenized_sentences, embedding_word_dict):
        words_train = []

        for sentence in tqdm.tqdm(tokenized_sentences):
            current_words = []
            for word_index in sentence:
                word = self.id2word[word_index]
                word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
                current_words.append(word_id)

            if len(current_words) >= self.pcfg.max_sentence_len:
                current_words = current_words[:self.pcfg.max_sentence_len]
            else:
                current_words += [len(embedding_word_dict) - 1] * (self.pcfg.max_sentence_len - len(current_words))
            words_train.append(current_words)
        return words_train

    def prepare_embeddings(self, words_dict, mode = 'fasttext'):
        print("Loading embeddings...")

        if mode == 'fasttext':
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec', binary=False)
        else:
            model = None
        embedding_word_dict = {w:ind for ind,w in enumerate(model.index2word)}
        embedding_size = 300

        print("Preparing data...")
        embedding_list, embedding_word_dict = self.clear_embedding_list(model, embedding_word_dict, words_dict)

        del model

        embedding_word_dict[self.cfg.unknown_word] = len(embedding_word_dict)
        embedding_list.append([0.] * embedding_size)
        embedding_word_dict[self.cfg.end_word] = len(embedding_word_dict)
        embedding_list.append([-1.] * embedding_size)

        embedding_matrix = np.array(embedding_list)


        id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())
        return embedding_matrix, embedding_word_dict, id_to_embedded_word

    def fit_tokenizer(self):
        print("Tokenizing sentences in train set...")
        tokenized_sentences_train, self.words_dict = self.tokenize_sentences(self.sentences_train, {})

        print("Tokenizing sentences in test set...")
        tokenized_sentences_test, self.words_dict = self.tokenize_sentences(self.sentences_test, self.words_dict)

        self.words_dict[self.cfg.unknown_word] = len(self.words_dict)
        self.id2word = dict((id, word) for word, id in self.words_dict.items())

        return tokenized_sentences_train, tokenized_sentences_test





class Model:

    def __init__(self, ModelConfig, Config):
        self.cfg = Config
        self.mcfg = ModelConfig()
        self.graph = tf.Graph()

    def set_graph(self, embedding_matrix):
        with self.graph.as_default():
            # tf Graph input
            tf.set_random_seed(1)

            self.x = tf.placeholder(tf.int32, shape=(None, self.mcfg.max_seq_len), name="input_x")
            self.y = tf.placeholder(tf.float32, shape=(None,6), name="input_y")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")

            self.logits = model_baseline(embedding_matrix,self.x,self.keep_prob)

            self.loss = binary_crossentropy(self.y,self.logits)
            self.cost = tf.losses.log_loss(predictions=self.logits, labels=self.y)
            #loss = tf.losses.sigmoid_cross_entropy(y,logits)
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss)
            (_, self.auc_update_op) = tf.contrib.metrics.streaming_auc(
                predictions=self.logits,
                labels=self.y,
                curve='ROC')


    def train(self, X_train, Y_train, X_valid, Y_valid, X_test, fold_id=0):
        sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
        train_iters = len(X_train) - self.mcfg.bsize
        steps = train_iters // self.mcfg.bsize
        valid_iters = len(X_valid) - self.mcfg.bsize
        with tf.Session(graph=self.graph) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(self.mcfg.epochs):
                tic = time.time()
                costs = []
                step = 0
                tf.initialize_local_variables().run(session=sess)
                while step * self.mcfg.bsize < train_iters:
                    batch_x = X_train[step * self.mcfg.bsize:(step + 1) * self.mcfg.bsize]
                    batch_y = Y_train[step * self.mcfg.bsize:(step + 1) * self.mcfg.bsize]
                    cost_ , _, roc_auc_train = sess.run([self.cost,self.optimizer,self.auc_update_op],feed_dict={self.x:batch_x,
                                                                     self.y:batch_y,
                                                                     self.keep_prob:0.7})
                    if step % 10 == 0:
                        print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,self.mcfg.epochs,step,steps,cost_))
                    costs.append(cost_)
                    step += 1

                vstep = 0
                vcosts = []
                vlosses = np.asarray([])
                tf.initialize_local_variables().run(session=sess)
                while vstep * self.mcfg.bsize < valid_iters:
                    test_cost_, valid_loss, roc_auc_test = sess.run([self.cost,self.loss,self.auc_update_op], feed_dict={self.x: X_valid[vstep * self.mcfg.bsize:(vstep + 1) * self.mcfg.bsize],
                                                           self.y: Y_valid[vstep * self.mcfg.bsize:(vstep + 1) * self.mcfg.bsize],
                                                           self.keep_prob: 1
                                                           })
                    vstep += 1
                    vcosts.append(test_cost_)
                    vlosses = np.concatenate((vlosses,valid_loss))
                avg_cost = np.log(np.mean(np.exp(vcosts)))
                toc = time.time()
                print('time needed %s' %(toc-tic))
                print('valid loss: %s' % avg_cost)
                print('roc auc test : {:.4}'.format(roc_auc_test))
                print('roc auc train : {:.4}'.format(roc_auc_train))
                avg_train_cost = np.log(np.mean(np.exp(costs[:valid_iters])))
                print('train loss %s' %avg_train_cost )

                self.populate_submission(X_test,sess,epoch, roc_auc_test,roc_auc_train, sample_submission,fold_id)


    def populate_submission(self,X_test,sess, epoch, roc_auc_test,roc_auc_train, sample_submission,fold_id):

        num_batches = (len(X_test) // self.mcfg.bsize) + 1
        res = np.zeros((len(X_test), 6))
        for s in range(num_batches):
            if s % 50 == 0:
                print(s)
            batch_x_test = X_test[s * self.mcfg.bsize:(s + 1) * self.mcfg.bsize]
            #if s == num_batches - 1:
            #    pad_size = bsize - batch_x_test.shape[0]
            #    pad = np.zeros(shape=(pad_size, X_test.shape[1]))
            #    batch_x_test = np.concatenate((batch_x_test, pad))
            logits_ = sess.run(self.logits, feed_dict={self.x: batch_x_test,
                                                  self.keep_prob: 1})
            #if s != num_batches - 1:
            res[s * self.mcfg.bsize:(s + 1) * self.mcfg.bsize] = logits_
            #else:
            #    res[s * bsize:(s + 1) * bsize - pad_size] = logits_[:bsize - pad_size]

        sample_submission[self.cfg.list_classes] = res

        dir_name = self.mcfg.model_name
        if not os.path.exists('submissions/' + dir_name):
            os.mkdir('submissions/' + dir_name)
        fn = "submissions/"
        fn += dir_name + "model_k"+ str(fold_id)
        fn += 'e' + str(epoch)

        fn += "v"+ str(round(roc_auc_test,ndigits=4))
        fn += "t"+ str(round(roc_auc_train,ndigits=4)) + ".csv"
        sample_submission.to_csv(fn, index=False)


cfg = Config()

tc = ToxicComments()
tokenized_sentences_train, tokenized_sentences_test = tc.fit_tokenizer()

embedding_matrix, embedding_word_dict, id_to_embedded_word = tc.prepare_embeddings(tc.words_dict)

train_list_of_token_ids = tc.convert_tokens_to_ids(tokenized_sentences_train,embedding_word_dict)
test_list_of_token_ids = tc.convert_tokens_to_ids(tokenized_sentences_test, embedding_word_dict)

X = np.array(train_list_of_token_ids)
Y = tc.train_data[tc.cfg.list_classes].values


fold_size = len(X) // 10
models = []

fold_start = 0
fold_end = fold_start + fold_size

X_valid = X[fold_start:fold_end]
Y_valid = Y[fold_start:fold_end]
X_train = np.concatenate([X[:fold_start], X[fold_end:]])
Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])
X_test = np.array(test_list_of_token_ids)


def train_folds(fold_count=10):
    fold_size = len(X) // fold_count
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        X_valid = X[fold_start:fold_end]
        Y_valid = Y[fold_start:fold_end]
        X_train = np.concatenate([X[:fold_start], X[fold_end:]])
        Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])


        #X_train, Y_train = mixup( X_train, Y_train,2, 0.1, seed=43)

        m = Model(ModelConfig, Config)
        m.set_graph(embedding_matrix)
        m.train(X_train, Y_train, X_valid, Y_valid, X_test, fold_id)



m = Model(ModelConfig,Config)
m.set_graph(embedding_matrix)
m.train(X_train,Y_train,X_valid,Y_valid,X_test,fold_id)