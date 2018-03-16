import pickle
import numpy as np
import tensorflow as tf
from train_model import ToxicComments
import tqdm
import pandas as pd
import os
from utilities import coverage, load_config
from preprocess_utils import Preprocessor, preprocess
from global_variables import UNKNOWN_WORD, END_WORD, NAN_WORD, LIST_CLASSES, LIST_LOGITS, COMMENT, \
    VALID_SLIM_FILENAME, TRAIN_SLIM_FILENAME, TEST_FILENAME, SAMPLE_SUBMISSION_FILENAME


LEVEL = 'word'


TRAIN_DATA_FN = TRAIN_SLIM_FILENAME
TEST_DATA_FN = TEST_FILENAME
VALID_DATA_FN = VALID_SLIM_FILENAME

use_GPU = True
root = 'models/CNNRNN/'
model = 'cudnn_slim/'
model_fp = root + model
logs = root + model + 'logs/'
fn_words_dict = root + model + 'tc_words_dict.p'
fn_embedding_words_dict = root + model + 'embedding_word_dict.p'
do_submission = True


class Config:
    pass

cfg = Config()
load_config(cfg, model_fp)


tc = ToxicComments(cfg)
epochs = [fn.split('.ckpt')[0] for fn in os.listdir(logs) if fn.endswith('.meta')]
results = pd.read_csv(root + model + 'results.csv')

def _get_score():
    rocs = np.zeros(len(epochs))
    loglosses = np.zeros(len(epochs))
    for k,epoch in enumerate(epochs):
        row = results.loc[results['fold_id'] == float(epoch.split('_')[0][1:])]
        row = row.loc[row['epoch'] == float(epoch.split('_')[1][1:])]
        roc = row['roc_auc_v'].values
        logloss = row['cost_val'].values
        rocs[k] = roc
        loglosses[k] = logloss
    print('mean roc: %s' %(np.prod(rocs)**(1/len(rocs))))
    print('mean logloss: %s' %(np.prod(loglosses)**(1/len(loglosses))))
_get_score()


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


with open(fn_words_dict, 'rb') as f:
    words_dict = pickle.load(f)
with open(fn_embedding_words_dict, 'rb') as f:
    embedding_word_dict = pickle.load(f)

embedding_matrix = np.load(root + model + 'embedding.npy')
tc.id2word = dict((id, word) for word, id in words_dict.items())
test_data = None
valid_data = None
if TEST_DATA_FN is not None:
    test_data = pd.read_csv(TEST_DATA_FN)
    if cfg.do_preprocess:
        print('preprocessing test')
        test_data = preprocess(test_data)
    if cfg.level == 'word':
        X_test = transform_data(test_data)
    else:
        train_data = pd.read_csv(TRAIN_DATA_FN)
        if cfg.do_preprocess:
            print('preprocessing train')
            train_data = preprocess(train_data)
        preprocessor = Preprocessor(min_count_chars=10)

        sentences_train = train_data["comment_text"].fillna("_NAN_").values
        # sentences_train = [preprocessor.lower(text) for text in sentences_train]
        preprocessor.create_char_vocabulary(sentences_train)
        embedding_matrix = np.zeros((preprocessor.char_vocab_size, cfg.char_embedding_size))
        sentences_test = test_data["comment_text"].fillna("_NAN_").values
        X_test = preprocessor.char2seq(sentences_test, maxlen=2000)
        embedding_matrix = np.zeros((preprocessor.char_vocab_size, cfg.char_embedding_size))
if VALID_DATA_FN is not None:
    valid_data = pd.read_csv(VALID_DATA_FN, index_col=1)
    if cfg.do_preprocess:
        print('preprocessing valid')
        valid_data = preprocess(valid_data)
    if cfg.level == 'word':
        X_valid = transform_data(valid_data)



def predict(epoch, X):
    tf.reset_default_graph()
    num_batches = len(X) // cfg.bsize + 1
    bsize_last_batch = len(X) % (cfg.bsize * (num_batches - 1))

    gpu_config = None
    if not use_GPU:
        gpu_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.InteractiveSession(config=gpu_config)

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



def fold_predicts(predicts_list, bagging_method = 'gmean'):

    if bagging_method == 'gmean':
        test_predicts = np.ones(predicts_list[0].shape)
        for fold_predict in predicts_list:
            test_predicts *= fold_predict

        test_predicts **= (1. / len(predicts_list))
    else:
        test_predicts = np.zeros(predicts_list[0].shape)
        for fold_predict in predicts_list:
            test_predicts += fold_predict

        test_predicts /= len(predicts_list)
    return test_predicts


test_results_list = []
for epoch in epochs:
    test_results = predict(epoch, X_test)
    test_results_list.append(test_results)
test_predicts = fold_predicts(test_results_list)
submission = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)
submission[LIST_CLASSES] = test_predicts
submission.to_csv(root + model + "test_data_folded.csv", index=False)


valid_results_list = []
for epoch in epochs:
    valid_results = predict(epoch, X_valid)
    valid_results_list.append(valid_results)


valid_predicts = fold_predicts(valid_results_list)
l2_valid_data = valid_data.copy()

l2_valid_data = pd.DataFrame(columns=LIST_LOGITS, index=l2_valid_data.index)
l2_valid_data[LIST_LOGITS] = valid_predicts
l2_valid_data.to_csv(root + model + "l2_valid_data.csv", index=False)





