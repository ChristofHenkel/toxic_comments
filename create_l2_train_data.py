import pickle
import numpy as np
import tensorflow as tf
from train_model import ToxicComments
import tqdm
import pandas as pd
import os
from utilities import coverage
from preprocess_utils import Preprocessor, preprocess
from global_variables import UNKNOWN_WORD, END_WORD, NAN_WORD, LIST_CLASSES, LIST_LOGITS, COMMENT, MODELS_FP


LEVEL = 'word'



TRAIN_DATA_FN = "assets/raw_data/bagging_train.csv"



type_ = 'RNN/'
model = 'pavel_all_outs_slim/'

model_fp = MODELS_FP+type_ +model

logs = model_fp+'logs/'
fn_words_dict = model_fp + 'tc_words_dict.p'
fn_embedding_words_dict = model_fp + 'embedding_word_dict.p'

class Config:
    max_seq_len = 500
    max_sentence_len = 500
    bsize = 512
    do_preprocess = True

cfg = Config()
tc = ToxicComments(Config)
epochs = [fn.split('.ckpt')[0] for fn in os.listdir(logs) if fn.endswith('.meta')]

with open(fn_words_dict, 'rb') as f:
    words_dict = pickle.load(f)
with open(fn_embedding_words_dict, 'rb') as f:
    embedding_word_dict = pickle.load(f)

embedding_matrix = np.load(model_fp + 'embedding.npy')
tc.id2word = dict((id, word) for word, id in words_dict.items())

def transform_data(data):
    if cfg.do_preprocess:
        data = preprocess(data)
    sentences = data["comment_text"].fillna("_NAN_").values
    # update word dict
    tokenized_sentences, _ = tc.tokenize_sentences(sentences, words_dict)
    coverage(tokenized_sentences,embedding_word_dict)
    sequences = tc.tokenized_sentences2seq(tokenized_sentences, words_dict)
    list_of_token_ids = tc.convert_tokens_to_ids(sequences, embedding_word_dict)
    X = np.array(list_of_token_ids)
    return X


train_data = pd.read_csv(TRAIN_DATA_FN)
X = transform_data(train_data)
Y = train_data[LIST_CLASSES].values
if LEVEL == 'char':
    preprocessor = Preprocessor(min_count_chars=20)
    sentences_train = train_data["comment_text"].fillna("_NAN_").values
    # sentences_train = [preprocessor.lower(text) for text in sentences_train]
    preprocessor.create_char_vocabulary(sentences_train)
    X = preprocessor.char2seq(sentences_train, maxlen=1000)


def predict(epoch, X):
    tf.reset_default_graph()
    num_batches = len(X) // cfg.bsize + 1
    bsize_last_batch = len(X) % (cfg.bsize * (num_batches - 1))
    sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))

    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(logs + epoch + '.ckpt.meta')
    #saver = tf.train.import_meta_graph(logs + 'k0_e3' + '.ckpt.meta')
    saver.restore(sess,logs + epoch + '.ckpt')
    #saver.restore(sess, logs + 'k0_e3' + '.ckpt')

    results = []
    #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for b in tqdm.tqdm(range(num_batches-1)):
        batch_x = X[b*cfg.bsize:(b+1)*cfg.bsize]
        result = sess.run('fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                                    'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result)

    if bsize_last_batch > 0:
        batch_x = X[(num_batches-1) * cfg.bsize:num_batches * cfg.bsize]
        b = cfg.bsize // bsize_last_batch + 1
        batch_x = np.repeat(batch_x, b, axis=0)
        batch_x = batch_x[:cfg.bsize]

        result = sess.run('fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                                    'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result[:bsize_last_batch])
    sess.close()
    del saver
    del sess
    results = np.concatenate(results, axis=0 )

    return results


fold_count=10

fold_size = len(X) // 10


res_X = np.zeros((len(X),6))
res_Y = np.zeros((len(X),6))

for epoch in epochs:
    fold_id = int(epoch[1])

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    results = predict(epoch, X_valid)
    res_X[fold_start:fold_end] = results
    res_Y[fold_start:fold_end] = Y_valid


l2_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
l2_data[LIST_LOGITS] = pd.DataFrame(res_X)
l2_data[LIST_CLASSES] = pd.DataFrame(res_Y)
l2_data.to_csv(model_fp + 'l2_train_data.csv')








