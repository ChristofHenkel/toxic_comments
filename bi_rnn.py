

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from utils import create_embedding_matrix
# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.contrib import layers
#from keras.models import Model
#from keras.layers import Dense, Embedding, Input
#from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.contrib.keras.api.keras.preprocessing import text, sequence
#from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer

max_features = 20000
maxlen = 100


train = pd.read_csv("assets/raw_data/train.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("Nothing").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_train = train[list_classes].values


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
split_at = int(len(list_tokenized_train)*0.9)
list_tokenized_test = list_tokenized_train[split_at:]
list_tokenized_train = list_tokenized_train[:split_at]

Y_test = Y_train[split_at:]
Y_train = Y_train[:split_at]

X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

pre_embedding = create_embedding_matrix(X_train,tokenizer.word_index)

# def words2seq(words,word2id)
graph = tf.Graph()

with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.int32, shape=(None,100), name="input_x")
        y = tf.placeholder(tf.int32, shape=(None,6), name="input_y")

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [max_features, 100], dtype=tf.float32,initializer=tf.constant_initializer(pre_embedding), trainable=False)
        #embedding = tf.get_variable("embedding", [max_features, 100], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
        # Creates Tensor with TensorShape([Dimension(batchsize), Dimension(nsteps), Dimension(embed_size)])

    with tf.name_scope("RNN"):

        #fw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
        #fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
        #bw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
        #bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)

        with tf.variable_scope('forward'):
            stacked_fw_rnn = []
            for fw_Lyr in range(1):
                fw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)  # or True
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
                stacked_fw_rnn.append(fw_cell)
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

        with tf.variable_scope('backward'):
            stacked_bw_rnn = []
            for bw_Lyr in range(1):
                bw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)  # or True
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedded_input,dtype=tf.float32)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
        output_fw, output_bw = outputs

        outputs = tf.add(output_fw,output_bw)
        #tf.reduce_mean(x, [1, 2])
        #tf.reduce_max()
        #outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        # flatten
        outputs = tf.contrib.layers.flatten(outputs)

        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)
        x3 = layers.dropout(x3, keep_prob=0.8)

        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        #tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
        #xent = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        #cost = tf.reduce_mean(xent, name='xent')
        cost = tf.losses.log_loss(predictions=logits, labels=y)

    gradients = tf.gradients(cost, tf.trainable_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).apply_gradients(
        zip(gradients, tf.trainable_variables()))

bsize = 512
train_iters = 84000
epochs = 3
steps = train_iters // bsize

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):

        step = 0
        while step * bsize < train_iters:
            batch_x = X_train[step*bsize:(step+1)*bsize]
            batch_y = Y_train[step*bsize:(step+1)*bsize]

            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,y:batch_y})
            print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))

            step += 1

        test_cost_ = sess.run(cost, feed_dict={x: X_test, y: Y_test})
        print('test loss: %s' %test_cost_)

def debug():
    with tf.Session(graph=graph) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        logits_ = sess.run(logits, feed_dict={x: batch_x, y: batch_y})