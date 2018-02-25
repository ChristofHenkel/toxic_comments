import numpy
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from sklearn.metrics import log_loss
import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import scipy
from utilities import corr_matrix
from global_variables import LIST_CLASSES, LIST_LOGITS



csvs_train = ['models/CNN/inception2_slim/l2_valid_data.csv',
              'models/NBSVM/slim/nbsvm_prediction_valid.csv',
              ]

dfs = [pd.read_csv(csv) for csv in csvs_train]
xs = [df[LIST_LOGITS].values for df in dfs]
n_models = len(csvs_train)

print('Corr matrix')
print(corr_matrix(xs))
print(' ')

for df in dfs:
    print(roc_auc_score(y_true=df[LIST_CLASSES].values, y_score=df[LIST_LOGITS].values))


ys = [df[LIST_CLASSES].values for df in dfs]

for i,_ in enumerate(csvs_train[1:]):
    assert np.array_equal(ys[0],ys[i])

X = np.concatenate([xs])
X = X.transpose([1,0,2])
Y = ys[0]

split_at = len(X)//10

kf = KFold(n_splits=10)

def squash(vector,epsilon=1e-9):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''

    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)

def routing(input, b_IJ, iter_routing=3, caps_dim_in=6, caps_dim_out=8, num_caps_out=6):
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
    num_caps_in = input.get_shape()[1]
    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, num_caps_in, num_caps_out, caps_dim_in, caps_dim_out), dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, num_caps_out, 1, 1])
    W = tf.tile(W, [bsize, 1, 1, 1, 1])
    assert input.get_shape() == [bsize, num_caps_in, num_caps_out, caps_dim_in, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [bsize, num_caps_in, num_caps_out, caps_dim_out, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [bsize, 1, num_caps_out, caps_dim_out, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [bsize, 1, num_caps_out, caps_dim_out, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, num_caps_in, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [bsize, num_caps_in, num_caps_out, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return (v_J)

bsize = 512
epochs = 50

for train, valid in kf.split(X):
    X_train = X[train]
    Y_train = Y[train]
    X_valid = X[valid]
    Y_valid = Y[valid]

    #X_train = X[split_at:]
    #Y_train = Y[split_at:]
    #X_valid = X[:split_at]
    #Y_valid = Y[:split_at]

    tf.reset_default_graph()
    graph = tf.Graph()
    n_caps1 = 3
    #cap_filter = 8
    cap_filter = 1
    with graph.as_default():

        x = tf.placeholder(shape=(bsize,n_models,len(LIST_CLASSES)), dtype=tf.float32)
        y = tf.placeholder(shape=(bsize,6),dtype=tf.float32)

        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(x, filters=cap_filter * n_caps1,
                                        kernel_size=1,
                                        strides=1,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 3 * cap_filter, n_caps1, 1))

            capsules = squash(capsules)

        input = tf.reshape(capsules, shape=(bsize, -1, 1, n_caps1, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            # about the reason of using 'batch_size', see issue #21
            b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 6, 1, 1], dtype=np.float32))
            capsules = routing(input, b_IJ, num_caps_out=6, caps_dim_out=8, caps_dim_in=3)
            capsules = tf.squeeze(capsules, axis=1)

        #cap_norms = tf.norm(capsules, axis=2)[:,:,0]
        #cap_norms = tf.minimum(tf.maximum(cap_norms,0),1)

        flat_capsules = tf.layers.flatten(capsules)

        logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)


        #logits = layers.fully_connected(h1,6,activation_fn=tf.nn.sigmoid)

        cost = tf.losses.log_loss(predictions=logits, labels=y)
        loss = binary_crossentropy(y, logits)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)



    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):
            step = 0
            while step * bsize < len(X_train) - bsize:
                batch_x = X_train[step * bsize:(step + 1) * bsize]
                batch_y = Y_train[step * bsize:(step + 1) * bsize]
                _ , logloss = sess.run([optimizer,cost], feed_dict={x:batch_x,y:batch_y})
                step += 1

            #logloss_val, logits_val = sess.run([cost,logits], feed_dict={x: X_valid, y: Y_valid})
                print(logloss)



        """
        def lloss(y_true,y_pred):
            l = 0
            for i in range(6):
                l += log_loss(y_true=y_true[:,i],y_pred=y_pred[:,i])
                l /= 6
            return l
    
        for x in xs:
            print(lloss(y_true=Y_valid,y_pred=x[valid]))
    
        print('------------->')
        print(lloss(y_true=Y_valid,y_pred=logits_val))
        print('using mean %s' %lloss(Y_valid,np.mean([x[valid] for x in xs],axis=0)))
        print('using geomean %s' % lloss(Y_valid, scipy.stats.gmean([x[valid] for x in xs], axis=0)))
        print('----------------------------')
        for x in xs:
            print(roc_auc_score(y_true=Y_valid,y_score=x[valid]))
        print('------------->')
        print(roc_auc_score(y_true=Y_valid,y_score=logits_val))
        print('using mean %s' %roc_auc_score(Y_valid,np.mean([x[valid] for x in xs],axis=0)))
        print('using geomean %s' % roc_auc_score(Y_valid, scipy.stats.gmean([x[valid] for x in xs], axis=0)))
        """