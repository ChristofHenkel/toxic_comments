import pandas as pd
from preprocess_utils import Preprocessor
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import numpy as np

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 2000

train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values
Y = train_data[list_classes].values

preprocessor = Preprocessor(min_count_chars=20)


sentences_train = [preprocessor.lower(text) for text in sentences_train]
preprocessor.create_char_vocabulary(sentences_train)
X = preprocessor.char2seq(sentences_train, maxlen=maxlen)

split_at = int(len(X) * 0.1)
X_valid = X[:split_at]
Y_valid = Y[:split_at]
X_train = X[split_at:]
Y_train = Y[split_at:]

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.int32,shape=(None,maxlen))
    y = tf.placeholder(dtype=tf.float32,shape=(None,6))
    keep_prob = tf.placeholder(dtype=tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    embedding = tf.get_variable("embedding", [preprocessor.char_vocab_size, 300], dtype=tf.float32)
    embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    x2 = tf.layers.conv1d(embedded_input, filters=8, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis = 2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=8, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
    x2 = tf.layers.conv1d(x2, filters=16, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=16, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
    x2 = tf.layers.conv1d(x2, filters=32, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=32, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
    x2 = tf.layers.conv1d(x2, filters=64, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=64, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
    x2 = tf.layers.conv1d(x2, filters=128, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=128, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
    x2 = tf.layers.conv1d(x2, filters=256, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.conv1d(x2, filters=256, kernel_size=3, strides=1, activation=tf.nn.elu)
    #x2 = tf.layers.batch_normalization(x2, axis=2, training=is_training)
    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

    fw_cell = tf.nn.rnn_cell.GRUCell(64)
    bw_cell = tf.nn.rnn_cell.GRUCell(64)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x2, dtype=tf.float32)
    output_fw, output_bw = outputs
    outputs = tf.concat([output_fw, output_bw], axis = 2)

    outputs = tf.transpose(outputs, [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)
    #x2 = tf.layers.flatten(x2)
    #x2 = tf.contrib.layers.fully_connected(x2, 1024, activation_fn=tf.nn.relu)
    #x2 = tf.contrib.layers.fully_connected(x2, 256, activation_fn=tf.nn.relu)
    x2 = tf.contrib.layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)

    loss = binary_crossentropy(y, logits)
    cost = tf.losses.log_loss(labels=y,predictions=logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

bsize = 512
train_iters = len(X_train) - bsize
valid_iters = len(X_valid) - bsize
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(15):
        step = 0
        while step * bsize < train_iters:
            batch_x = X_train[step * bsize:(step + 1) * bsize]
            batch_y = Y_train[step * bsize:(step + 1) * bsize]
            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:0.7,
                                                             is_training:True})

            print('e%s -- s%s -- cost: %s' %(epoch,step,cost_))
            step +=1

        vstep = 0
        vcosts = []
        while vstep * bsize < valid_iters:
            test_cost_ = sess.run(cost, feed_dict={x: X_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   y: Y_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   keep_prob: 1,
                                                   is_training: False
                                                   })
            vstep += 1
            vcosts.append(test_cost_)
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        print('valid loss: %s' % avg_cost)
