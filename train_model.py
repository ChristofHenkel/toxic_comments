import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf
from collections import Counter
from utilities import get_oov_vector
import nltk
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors, FastText
import tqdm
import os
import time
from preprocess_utils import Preprocessor
from augmentation import retranslation, mixup, synonyms
from architectures import CNN
import pickle
from utilities import loadGloveModel, coverage


model_baseline = CNN().inception_3
unknown_word = "_UNK_"
end_word = "_END_"
nan_word = "_NAN_"
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
results = pd.DataFrame(columns=['fold_id','epoch','roc_auc_v','roc_auc_t','cost_val'])

train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values

class Config:

    max_sentence_len = 500
    do_augmentation_with_translate = False
    do_augmentation_with_mixup = False
    do_synthezize_embeddings = False
    mode_embeddings = 'fasttext_300d'
    if do_synthezize_embeddings:
        synth_threshold = 0.7
    bsize = 512
    max_seq_len = 500
    epochs = 20
    model_name = 'inception_2_3'
    root = ''
    fp = 'models/CNN/' + model_name + '/'
    logs_path = fp + 'logs/'
    if not os.path.exists(root + fp):
        os.mkdir(root + fp)
    max_models_to_keep = 1
    save_by_roc = False

    lr = 0.0005
    keep_prob = 0.8

class ToxicComments:

    def __init__(self,Config):
        self.preprocessor = Preprocessor()
        self.cfg = Config()
        self.word_counter = Counter()
        self.words_dict = {}

    def tokenize_sentences(self,sentences, words_dict, mode = 'twitter'):
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
            self.word_counter.update(tokens)
            for word in tokens:
                self.word_counter.update([word])
                word = word.lower()
                if word not in words_dict:
                    words_dict[word] = len(words_dict)
                result.append(word)
            tokenized_sentences.append(result)
        return tokenized_sentences, words_dict

    def tokenized_sentences2seq(self,tokenized_sentences, words_dict):
        sequences = []
        for sentence in tqdm.tqdm(tokenized_sentences, mininterval=5):
            seq = []
            for token in sentence:
                try:
                    seq.append(words_dict[token])
                except KeyError:
                    seq.append(words_dict[unknown_word])
            sequences.append(seq)
        return sequences

    def update_words_dict(self,tokenized_sentences):
        self.words_dict.pop(unknown_word, None)
        k = 0
        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in self.words_dict:
                    k += 1
                    self.words_dict[token] = len(self.words_dict)
        print('{} words added'.format(k))
        self.words_dict[unknown_word] = len(self.words_dict)
        self.id2word = dict((id, word) for word, id in self.words_dict.items())

    def clear_embedding_list(self,model, embedding_word_dict, words_dict):
        cleared_embedding_list = []
        cleared_embedding_word_dict = {}
        k = 0
        l = 0
        for word in tqdm.tqdm(words_dict):
            if word not in embedding_word_dict:
                l += 1
                if self.cfg.do_synthezize_embeddings:

                    row = get_oov_vector(word, model, threshold=self.cfg.synth_threshold)
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

        for sentence in tqdm.tqdm(tokenized_sentences):
            current_words = []
            for word_index in sentence:
                word = self.id2word[word_index]
                word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
                current_words.append(word_id)

            if len(current_words) >= self.cfg.max_sentence_len:
                current_words = current_words[:self.cfg.max_sentence_len]
            else:
                current_words += [len(embedding_word_dict) - 1] * (self.cfg.max_sentence_len - len(current_words))
            words_train.append(current_words)
        return words_train

    def prepare_embeddings(self, words_dict):
        print("Loading embeddings...")

        if self.cfg.mode_embeddings == 'fasttext_300d':
            print('loading Fasttext 300d')
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec', binary=False)
            embedding_word_dict = {w: ind for ind, w in enumerate(model.index2word)}
        elif self.cfg.mode_embeddings == 'mini_fasttext_300d':
            model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/mini_fasttext_300d2.vec',binary=False)
            embedding_word_dict = {w: ind for ind, w in enumerate(model.index2word)}
        elif self.cfg.mode_embeddings == 'fasttext_wiki_300d':
            model = FastText.load_fasttext_format('assets/embedding_models/ft_wiki/wiki.en.bin')
            embedding_word_dict = {w: ind for ind, w in enumerate(model.wv.index2word)}
        elif self.cfg.mode_embeddings == 'glove_300d':
            model = loadGloveModel('assets/embedding_models/glove/glove.840B.300d.txt',dims=300)
            embedding_word_dict = {w: ind for ind, w in enumerate(model)}
        else:
            model = None


        embedding_size = 300

        print("Preparing data...")
        if not self.cfg.mode_embeddings == 'fasttext_wiki_300d':
            embedding_list, embedding_word_dict = self.clear_embedding_list(model, embedding_word_dict, words_dict)
        else:
            embedding_list, embedding_word_dict = self.clear_embedding_list_fasttext(model, words_dict)

        del model

        embedding_word_dict[unknown_word] = len(embedding_word_dict)
        embedding_list.append([0.] * embedding_size)
        embedding_word_dict[end_word] = len(embedding_word_dict)
        embedding_list.append([-1.] * embedding_size)

        embedding_matrix = np.array(embedding_list)


        id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())
        return embedding_matrix, embedding_word_dict, id_to_embedded_word

    def fit_tokenizer(self,list_of_sentences):

        list_of_tokenized_sentences = []
        for sentences in list_of_sentences:
            tokenized_sentences, self.words_dict = self.tokenize_sentences(sentences, self.words_dict)
            list_of_tokenized_sentences.append(tokenized_sentences)

        self.words_dict[unknown_word] = len(self.words_dict)
        self.id2word = dict((id, word) for word, id in self.words_dict.items())

        return list_of_tokenized_sentences

    def save(self):
        with open(self.cfg.fp + 'tc.p','wb') as f:
            pickle.dump(self,f)


class Model:

    def __init__(self, Config):
        self.cfg = Config
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

            self.x = tf.placeholder(tf.int32, shape=(None, self.cfg.max_seq_len), name="x")
            self.y = tf.placeholder(tf.float32, shape=(None,6), name="y")
            self.em = tf.placeholder(tf.float32, shape=(embedding_matrix.shape[0], embedding_matrix.shape[1]), name="em")
            self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

            self.output = model_baseline(self.em,self.x,self.keep_prob)


            with tf.variable_scope('logits'):
                self.logits = self.output

            with tf.variable_scope('loss'):
                self.loss = binary_crossentropy(self.y,self.logits)
                self.cost = tf.losses.log_loss(predictions=self.logits, labels=self.y)
                (_, self.auc_update_op) = tf.metrics.auc(predictions=self.logits,labels=self.y,curve='ROC')

            with tf.variable_scope('optim'):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.cfg.lr).minimize(self.loss)

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
                                                                   self.em:embedding_matrix,
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
                    test_cost_, valid_loss, roc_auc_valid = sess.run([self.cost,self.loss,self.auc_update_op],
                                                                    feed_dict={self.x: batch_x_valid,
                                                           self.y: batch_y_valid,
                                                        self.em: embedding_matrix,
                                                           self.keep_prob: 1
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

        sample_submission[list_classes] = res

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

    tc = ToxicComments(Config)

    Y = train_data[list_classes].values

    tokenized_sentences_train, tokenized_sentences_test = tc.fit_tokenizer([sentences_train,sentences_test])


    with open(tc.cfg.fp + 'tc_words_dict.p','wb') as f:
        pickle.dump(tc.words_dict,f)

    sequences_train = tc.tokenized_sentences2seq(tokenized_sentences_train, tc.words_dict)
    #sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, tc.words_dict)
    embedding_matrix, embedding_word_dict, id_to_embedded_word = tc.prepare_embeddings(tc.words_dict)
    coverage(tokenized_sentences_train,embedding_word_dict)
    with open(tc.cfg.fp + 'embedding_word_dict.p','wb') as f:
        pickle.dump(embedding_word_dict,f)
    np.save(tc.cfg.fp + 'embedding.npy',embedding_matrix)
    train_list_of_token_ids = tc.convert_tokens_to_ids(sequences_train, embedding_word_dict)
    #test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)

    X = np.array(train_list_of_token_ids)
    #X_test = np.array(test_list_of_token_ids)
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


        #X_train, Y_train = mixup( X_train, Y_train,2, 0.1, seed=43)

        m = Model(Config)
        m.set_graph(embedding_matrix)
        m.train(X_train, Y_train, X_valid, Y_valid, X_test, embedding_matrix, fold_id)

def predict(sentences_test=None, X_test=None, do_submission = False):
    bsize = 512
    type_ = 'models/CNN/'
    model = 'vgg_4/'
    epoch = 'k0_e3'
    logs = type_ + model + 'logs/' + epoch

    if sentences_test is not None:
        with open(Config().fp + 'tc.p','wb') as f:
            tc = pickle.load(f)
        tokenized_sentences_test = tc.tokenize_sentences(sentences_test, tc.words_dict)
        sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, tc.words_dict)
        test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)
        X_test = np.array(test_list_of_token_ids)

    num_batches = len(X_test) // bsize + 1
    bsize_last_batch = len(X_test) % (num_batches - 1)
    sess = tf.InteractiveSession()

    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(logs + '.ckpt.meta')
    saver.restore(sess,logs + '.ckpt')

    results = []
    #[n.name for n in tf.get_default_graph().as_graph_def().node]
    for b in tqdm.tqdm(range(num_batches-1)):
        batch_x = X_test[b*bsize:(b+1)*bsize]
        result = sess.run('FCCaps_layer/fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                              'keep_prob:0': 1})
        results.append(result)

    if bsize_last_batch > 0:
        batch_x = X_test[(num_batches-1) * bsize:num_batches * bsize]
        b = bsize // bsize_last_batch + 1
        batch_x = np.repeat(batch_x, b, axis=0)
        batch_x = batch_x[:bsize]

        result = sess.run('FCCaps_layer/fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                              'keep_prob:0': 1})
        results.append(result[:bsize_last_batch])


    results = np.concatenate( results, axis=0 )

    if do_submission:
        sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
        sample_submission[list_classes] = results
        if not os.path.exists(type_ + model + 'submissions/'):
            os.mkdir(type_ + model + 'submissions/')

        fn = type_ + model + 'submissions/'
        fn += model[:-1] + epoch + '.csv'
        sample_submission.to_csv(fn, index=False)
    return results