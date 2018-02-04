"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf
from caps_config import cfg
from capsLayer import CapsLayer

import pandas as pd
from preprocess_utils import Preprocessor
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import numpy as np
import os

def routing(input, b_IJ):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    bsize = input.get_shape()[0]
    num_caps = input.get_shape()[1]
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [bsize, 1, 1, 1, 1])
    assert input.get_shape() == [bsize, num_caps, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [bsize, num_caps, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [bsize, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [bsize, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, num_caps, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [bsize, num_caps, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

epsilon = 1e-9
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 1000
DEPTH = 7
EPOCHS = 15
bsize = 128
model_name = 'vgg7_3'
root = '/home/christof/kaggle/toxic_comments/'
fp = 'models/CAPS/' + model_name + '/'
logs_path = fp + 'logs/'
if not os.path.exists(root + fp):
    os.mkdir(root + fp)


results = pd.DataFrame(columns=['e','roc_auc_v','roc_auc_t'])




train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values
Y = train_data[list_classes].values

preprocessor = Preprocessor(min_count_chars=20)


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

    with tf.variable_scope('Input'):
        x = tf.placeholder(dtype=tf.int32,shape=(None,maxlen))
        y = tf.placeholder(dtype=tf.float32,shape=(None,6))

        logits =

        loss = binary_crossentropy(y, logits)
        cost = tf.losses.log_loss(labels=y, predictions=logits)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)
        (_, auc_update_op) = tf.contrib.metrics.streaming_auc(
            predictions=logits,
            labels=y,
            curve='ROC')


train_iters = len(X_train) - 2*bsize
valid_iters = len(X_valid) - 2*bsize

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
                                                             y:batch_y})

            print('e%s -- s%s -- cost: %s' %(epoch,step,cost_))

            step +=1

        vstep = 0
        vcosts = []
        tf.local_variables_initializer().run(session=sess)
        conv_out_valid = np.zeros((len(X_valid), 32))
        while vstep * bsize < valid_iters:
            test_cost_, roc_auc_test = sess.run([cost,auc_update_op], feed_dict={x: X_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   y: Y_valid[vstep * bsize:(vstep + 1) * bsize]})

            vcosts.append(test_cost_)

            vstep += 1
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        print('valid loss: %s' % avg_cost)
        print('roc auc test : {:.4}'.format(roc_auc_test))
        print('roc auc train : {:.4}'.format(roc_auc_train))
        #results.loc[len(results)] = [epoch, roc_auc_test, roc_auc_train]
        #results.to_csv(fp + 'results.csv')
