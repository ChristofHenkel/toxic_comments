import tensorflow as tf
from tensorflow.contrib import layers

def pavel(embedding_matrix, x, keep_prob):

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('forward'):

        fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        fw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        stacked_fw_rnn = [fw_cell1,fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        bw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        stacked_bw_rnn = [bw_cell1,bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.concat([output_fw, output_bw], axis = 2)

    outputs = tf.transpose(outputs, [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)
    #outputs = outputs[:,:,-1]

    x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
    return logits

def pavel2(embedding_matrix, x, keep_prob):
    """
    like pavel but with different cell
    :param embedding_matrix:
    :param x:
    :param keep_prob:
    :return:
    """

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('forward'):

        fw_cell1 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        fw_cell2 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
        #fw_cell2 = tf.nn.rnn_cell.DropoutWrapper(fw_cell2, output_keep_prob=keep_prob)
        stacked_fw_rnn = [fw_cell1,fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        bw_cell2 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
        #bw_cell2 = tf.nn.rnn_cell.DropoutWrapper(bw_cell2, output_keep_prob=keep_prob)
        stacked_bw_rnn = [bw_cell1,bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.concat([output_fw, output_bw], axis = 2)
    outputs = tf.transpose(outputs, [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)
    #outputs = outputs[:,:,-1]

    x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
    return logits

def rnn_3(embedding_matrix, x, keep_prob):

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('forward'):

        fw_cell1 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        fw_cell2 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        stacked_fw_rnn = [fw_cell1,fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        bw_cell2 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        stacked_bw_rnn = [bw_cell1,bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.concat([output_fw, output_bw], axis = 2)

    outputs = tf.transpose(outputs, [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)

    x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
    return logits

def rnn_4(embedding_matrix, x, keep_prob):

    with tf.name_scope("Embedding"):
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('forward'):

        fw_cell1 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
        fw_cell2 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        stacked_fw_rnn = [fw_cell1,fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
        bw_cell2 = tf.nn.rnn_cell.GRUCell(64,activation=tf.nn.elu)
        stacked_bw_rnn = [bw_cell1,bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.concat([output_fw, output_bw], axis = 2)

    outputs = tf.transpose(outputs, [0, 2, 1])

    outputs = tf.reduce_max(outputs, axis=2)

    x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
    return logits