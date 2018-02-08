import pickle
import numpy as np
import tensorflow as tf
from train_model import ToxicComments
import tqdm
import pandas as pd
import os
from utilities import coverage

unknown_word = "_UNK_"
end_word = "_END_"
nan_word = "_NAN_"
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_test = test_data["comment_text"].fillna("_NAN_").values

bsize = 512
type_ = 'models/CNN/'
model = 'inception_2_2/'

logs = type_ + model + 'logs/'
fn_words_dict = type_ + model + 'tc_words_dict.p'
fn_embedding_words_dict = type_ + model + 'embedding_word_dict.p'
do_submission = True

class Config:
    max_seq_len = 500
    max_sentence_len = 500

tc = ToxicComments(Config)
epochs = [fn.split('.ckpt')[0] for fn in os.listdir(logs) if fn.endswith('.meta')]
results = pd.read_csv(type_ + model + 'results.csv')

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

with open(fn_words_dict, 'rb') as f:
    words_dict = pickle.load(f)
with open(fn_embedding_words_dict, 'rb') as f:
    embedding_word_dict = pickle.load(f)
tokenized_sentences_test, _ = tc.tokenize_sentences(sentences_test, words_dict)
coverage(tokenized_sentences_test,embedding_word_dict)

embedding_matrix = np.load(type_ + model + 'embedding.npy')
sequences_test = tc.tokenized_sentences2seq(tokenized_sentences_test, words_dict)
tc.id2word = dict((id, word) for word, id in words_dict.items())

test_list_of_token_ids = tc.convert_tokens_to_ids(sequences_test, embedding_word_dict)
X_test = np.array(test_list_of_token_ids)

def predict(epoch, X_test, do_submission):

    num_batches = len(X_test) // bsize + 1
    bsize_last_batch = len(X_test) % (num_batches - 1)
    sess = tf.InteractiveSession()


        # load meta graph and restore weights
    saver = tf.train.import_meta_graph(logs + epoch + '.ckpt.meta')
    saver.restore(sess,logs + epoch + '.ckpt')

    results = []
    #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    for b in tqdm.tqdm(range(num_batches-1)):
        batch_x = X_test[b*bsize:(b+1)*bsize]
        result = sess.run('fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,
                                                                    'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result)

    if bsize_last_batch > 0:
        batch_x = X_test[(num_batches-1) * bsize:num_batches * bsize]
        b = bsize // bsize_last_batch + 1
        batch_x = np.repeat(batch_x, b, axis=0)
        batch_x = batch_x[:bsize]

        result = sess.run('fully_connected/Sigmoid:0', feed_dict={'x:0': batch_x,'em:0':embedding_matrix,
                                                              'keep_prob:0': 1})
        results.append(result[:bsize_last_batch])
    sess.close()
    del saver
    del sess
    results = np.concatenate( results, axis=0 )

    if do_submission:
        sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
        sample_submission[list_classes] = results
        if not os.path.exists(type_ + model + 'submissions/'):
            os.mkdir(type_ + model + 'submissions/')
        fn = type_ + model + 'submissions/'
        fn += model[:-1] + epoch + '.csv'
        sample_submission.to_csv(fn, index=False)
    #return results

for epoch in epochs:
    predict(epoch, X_test, do_submission)


csv_files = os.listdir(type_ + model + 'submissions/')

test_predicts_list = []
for csv_file in csv_files:
    orig_submission = pd.read_csv(type_ + model + 'submissions/' + csv_file)
    predictions = orig_submission[list_classes]
    test_predicts_list.append(predictions)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

#test_predicts = np.multiply(*test_predicts_list)
test_predicts **= (1. / len(test_predicts_list))

new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
new_submission[list_classes] = test_predicts
new_submission.to_csv(type_ + model + "folded.csv", index=False)