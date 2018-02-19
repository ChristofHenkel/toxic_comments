import pandas as pd
from preprocess_utils import Preprocessor
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from tensorflow.contrib import layers
import numpy as np
import os
import pickle
import tqdm

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 2000
EPOCHS = 15
model_name = 'crnn_1'
root = '/home/christof/kaggle/toxic_comments/'
fp = 'models/CRNN/' + model_name + '/'
logs_path = fp + 'logs/'
if not os.path.exists(root + fp):
    os.mkdir(root + fp)
results = pd.DataFrame(columns=['e', 'roc_auc_v', 'roc_auc_t'])






train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values
Y = train_data[list_classes].values

preprocessor = Preprocessor(min_count_chars=10)


#sentences_train = [preprocessor.lower(text) for text in sentences_train]
preprocessor.create_char_vocabulary(sentences_train)
X = preprocessor.char2seq(sentences_train, maxlen=maxlen)

split_at = int(len(X) * 0.1)
X_valid = X[:split_at]
Y_valid = Y[:split_at]
X_train = X[split_at:]
Y_train = Y[split_at:]

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.int32,shape=(None,maxlen),name='x')
    y = tf.placeholder(dtype=tf.float32,shape=(None,6),name='y')
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

    embedding = tf.get_variable("embedding", [preprocessor.char_vocab_size, 200], dtype=tf.float32)
    embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    x2 = embedded_input
    for i in range(3, 3 + 2):
        x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
        x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

    with tf.variable_scope('forward'):
        fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        fw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        stacked_fw_rnn = [fw_cell1, fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        bw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        stacked_bw_rnn = [bw_cell1, bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, x2, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.transpose(tf.concat([output_fw, output_bw], axis=2), [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)
    # outputs = outputs[:,:,-1]

    with tf.variable_scope('fc'):
        prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

    logits = tf.contrib.layers.fully_connected(prelogits, 6, activation_fn=tf.nn.sigmoid)

    loss = binary_crossentropy(y, logits)
    cost = tf.losses.log_loss(labels=y,predictions=logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
    (_, auc_update_op) = tf.contrib.metrics.streaming_auc(
        predictions=logits,
        labels=y,
        curve='ROC')
    saver = tf.train.Saver(max_to_keep=15)

def save_ckpt(saver, sess, epoch,roc_auc_test, roc_auc_train):
    print('saving model...', end='')
    model_name = 'e%s.ckpt' %epoch
    s_path = saver.save(sess, logs_path + model_name)
    print("Model saved in file: %s" % s_path)
    results.loc[len(results)] = [epoch, roc_auc_test, roc_auc_train]
    results.to_csv(fp + 'results.csv')

def train():
    with open(fp + 'preprocessor.p','wb') as f:
        pickle.dump(preprocessor,f)
    bsize = 512
    train_iters = len(X_train) - bsize
    valid_iters = len(X_valid) - bsize
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(EPOCHS):
            step = 0
            tf.local_variables_initializer().run(session=sess)
            while step * bsize < train_iters:
                batch_x = X_train[step * bsize:(step + 1) * bsize]
                batch_y = Y_train[step * bsize:(step + 1) * bsize]
                cost_ , _, roc_auc_train = sess.run([cost,optimizer,auc_update_op],feed_dict={x:batch_x,
                                                                 y:batch_y,
                                                                 keep_prob:0.7,
                                                                 })

                print('e%s -- s%s -- cost: %s' %(epoch,step,cost_))

                step +=1

            vstep = 0
            vcosts = []
            tf.local_variables_initializer().run(session=sess)
            while vstep * bsize < valid_iters:
                test_cost_, roc_auc_test = sess.run([cost,auc_update_op], feed_dict={x: X_valid[vstep * bsize:(vstep + 1) * bsize],
                                                       y: Y_valid[vstep * bsize:(vstep + 1) * bsize],
                                                       keep_prob: 1

                                                       })

                vcosts.append(test_cost_)
                vstep += 1
            avg_cost = np.log(np.mean(np.exp(vcosts)))
            print('valid loss: %s' % avg_cost)
            print('roc auc test : {:.4}'.format(roc_auc_test))
            print('roc auc train : {:.4}'.format(roc_auc_train))
            save_ckpt(saver, sess, epoch, roc_auc_test, roc_auc_train)


def predict(sentences):
    bsize = 512
    model = 'models/CRNN/crnn_1/'
    logs = model + 'logs/e14'
    with open(model + 'preprocessor.p','rb') as f:
        preprocessor = pickle.load(f)

    print('converting string to id-sequence')
    X = preprocessor.char2seq(sentences, maxlen=maxlen)

    num_batches = len(X) // bsize + 1
    sess = tf.InteractiveSession()

    # load meta graph and restore weights
    saver = tf.train.import_meta_graph(logs + '.ckpt.meta')
    saver.restore(sess,logs + '.ckpt')

    results = []
    #log_loss/value:0
    for b in tqdm.tqdm(range(num_batches)):
        batch_x = X[b*bsize:(b+1)*bsize]
        result = sess.run('fc/fully_connected/Elu:0', feed_dict={'x:0': batch_x,
                                                              'keep_prob:0': 1,
                                                              'is_training:0': False})
        results.append(result)
    results = np.concatenate( results, axis=0 )
    return results
train()