import pickle
import numpy as np
import tensorflow as tf
from train_model import ToxicComments
import tqdm
import pandas as pd
import os
from utilities import coverage
from preprocess_utils import Preprocessor, preprocess
from global_variables import UNKNOWN_WORD, END_WORD, NAN_WORD, LIST_CLASSES, LIST_LOGITS, COMMENT


LEVEL = 'word'
MODE = 'test'
TRAIN_DATA_FN = "assets/raw_data/bagging_train.csv"
#TEST_DATA_FN = "assets/raw_data/test.csv"
TEST_DATA_FN = 'assets/raw_data/bagging_valid.csv'

bsize = 512
type_ = 'models/RNN/'
model = 'pavel_attention_slim2/'

logs = type_ + model + 'logs/'
fn_words_dict = type_ + model + 'tc_words_dict.p'
fn_embedding_words_dict = type_ + model + 'embedding_word_dict.p'
do_submission = True
do_preprocess = True



class Config:
    max_seq_len = 500
    max_sentence_len = 500

tc = ToxicComments(Config)
epochs = [fn.split('.ckpt')[0] for fn in os.listdir(logs) if fn.endswith('.meta')]
results = pd.read_csv(type_ + model + 'results.csv')

def _get_score():
    rocs = np.zeros(10)
    loglosses = np.zeros(10)
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
with open(fn_words_dict, 'rb') as f:
    words_dict = pickle.load(f)
with open(fn_embedding_words_dict, 'rb') as f:
    embedding_word_dict = pickle.load(f)

embedding_matrix = np.load(type_ + model + 'embedding.npy')
tc.id2word = dict((id, word) for word, id in words_dict.items())

def transform_data(data):
    if do_preprocess:
        data = preprocess(data)
    sentences = data["comment_text"].fillna("_NAN_").values
    tokenized_sentences, _ = tc.tokenize_sentences(sentences, words_dict)
    coverage(tokenized_sentences,embedding_word_dict)
    sequences = tc.tokenized_sentences2seq(tokenized_sentences, words_dict)
    list_of_token_ids = tc.convert_tokens_to_ids(sequences, embedding_word_dict)
    X = np.array(list_of_token_ids)
    return X

if MODE == 'train':
    train_data = pd.read_csv(TRAIN_DATA_FN)
    X = transform_data(train_data)
    if LEVEL == 'char':
        preprocessor = Preprocessor(min_count_chars=20)
        sentences_train = train_data["comment_text"].fillna("_NAN_").values
        # sentences_train = [preprocessor.lower(text) for text in sentences_train]
        preprocessor.create_char_vocabulary(sentences_train)
        X = preprocessor.char2seq(sentences_train, maxlen=1000)
else:
    test_data = pd.read_csv(TEST_DATA_FN)
    X = transform_data(test_data)



def predict(epoch, X):
    tf.reset_default_graph()
    num_batches = len(X) // bsize + 1
    bsize_last_batch = len(X) % (bsize * (num_batches - 1))
    sess = tf.InteractiveSession()

    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(logs + epoch + '.ckpt.meta')
    saver.restore(sess,logs + epoch + '.ckpt')

    results = []
    #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for b in tqdm.tqdm(range(num_batches-1)):
        batch_x = X[b*bsize:(b+1)*bsize]
        result = sess.run('fully_connected_3/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                                    'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result)

    if bsize_last_batch > 0:
        batch_x = X[(num_batches-1) * bsize:num_batches * bsize]
        b = bsize // bsize_last_batch + 1
        batch_x = np.repeat(batch_x, b, axis=0)
        batch_x = batch_x[:bsize]

        result = sess.run('fully_connected_3/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                                    'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result[:bsize_last_batch])
    sess.close()
    del saver
    del sess
    results = np.concatenate(results, axis=0 )

    return results

def populate_submission(results):
    if MODE == 'train':
        submission_folder = 'train_logits/'
        submission = train_data.copy()
        submission.drop(columns = ["comment_text"])
        submission[LIST_LOGITS] = pd.DataFrame(results, index=submission.index)
    else:
        try:
            a = test_data['toxic']
            submission_folder = 'bagging_logits/'
            submission = test_data.copy()
            submission.drop(columns=["comment_text"])
            submission[LIST_LOGITS] = pd.DataFrame(results, index=submission.index)

        except KeyError:
            submission_folder = 'submissions/'
            submission = pd.read_csv("assets/raw_data/sample_submission.csv")
            submission[LIST_CLASSES] = results
    if not os.path.exists(type_ + model + submission_folder):
        os.mkdir(type_ + model + submission_folder)
    fn = type_ + model + submission_folder
    fn += model[:-1] + epoch + '.csv'
    submission.to_csv(fn, index=False)





def fold_submissions():
    if MODE == 'train':
        submission_folder = 'train_logits/'
    else:
        try:
            a = test_data['toxic']
            submission_folder = 'bagging_logits/'
        except KeyError:
            submission_folder = 'submissions/'
    csv_files = os.listdir(type_ + model + submission_folder)

    test_predicts_list = []
    for csv_file in csv_files:
        orig_submission = pd.read_csv(type_ + model + submission_folder + csv_file)
        if MODE == 'train':
            predictions = orig_submission[LIST_LOGITS]

        else:
            try:
                a = test_data['toxic']
                predictions = orig_submission[LIST_LOGITS]
            except KeyError:
                predictions = orig_submission[LIST_CLASSES]
        test_predicts_list.append(predictions)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))

    if MODE == 'train':
        new_submission = orig_submission.copy()
        new_submission[LIST_LOGITS]=test_predicts
        new_submission.to_csv(type_ + model + "train_logits_folded.csv", index=False)
    else:
        try:
            a = test_data['toxic']
            new_submission = orig_submission.copy()
            new_submission[LIST_LOGITS] = test_predicts
            new_submission.to_csv(type_ + model + "baggin_logits_folded.csv", index=False)
        except KeyError:

            new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
            new_submission[LIST_CLASSES] = test_predicts
            new_submission.to_csv(type_ + model + "folded.csv", index=False)


for epoch in epochs:
    results = predict(epoch, X)
    if do_submission:
        populate_submission(results)

fold_submissions()