import tensorflow as tf
from tensorflow.contrib import layers

def cnn_v1(x,vocab_size,nb_filter,filter_kernels,dense_outputs):
    """
    from https://medium.com/@surmenok/character-level-convolutional-networks-for-text-classification-d582c0c36ace
    :param x:
    :param vocab_size:
    :param nb_filter:
    :param filter_kernels:
    :param dense_outputs:
    :return:
    """
    embedding = tf.get_variable("embedding", [vocab_size, 21], dtype=tf.float32)
    embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    x2 = tf.layers.conv1d(embedded_input,filters=nb_filter,kernel_size=filter_kernels[0])
    x2 = tf.layers.max_pooling1d(x2,3,1)

    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[1],)
    x2 = tf.layers.max_pooling1d(x2,3,1)

    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[2])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[3])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[4])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[5])

    x2 = tf.layers.max_pooling1d(x2, 3, 1)
    x2 = tf.layers.flatten(x2)

    x2 = layers.fully_connected(x2,dense_outputs)
    x2 = layers.dropout(x2,keep_prob=0.5)
    x2 = layers.fully_connected(x2, dense_outputs)
    x2 = layers.dropout(x2, keep_prob=0.5)
    logits = layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)
    return logits

def cnn_rnn_v1(x,vocab_size,nb_filter,filter_kernels,dense_outputs):
    """
    https://offbit.github.io/how-to-read/
    :param x:
    :param vocab_size:
    :param nb_filter:
    :param filter_kernels:
    :param dense_outputs:
    :return:
    """

    embedding = tf.get_variable("embedding", [vocab_size, 64], dtype=tf.float32)
    embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    x2 = tf.layers.conv1d(embedded_input,filters=nb_filter,kernel_size=filter_kernels[0])
    x2 = tf.layers.max_pooling1d(x2,3,1)

    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[1])
    x2 = tf.layers.max_pooling1d(x2,3,1)

    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[2])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[3])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[4])
    x2 = tf.layers.conv1d(x2,filters=nb_filter,kernel_size=filter_kernels[5])

    x2 = tf.layers.max_pooling1d(x2, 3, 1)
    #x2 = tf.layers.flatten(x2)

    fw_cell = tf.contrib.rnn.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True)
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
    bw_cell = tf.contrib.rnn.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True)
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x2, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.add(output_fw, output_bw)
    outputs = tf.contrib.layers.flatten(outputs)

    x2 = layers.fully_connected(outputs, 256, activation_fn=tf.nn.elu)
    x2 = layers.fully_connected(x2,64, activation_fn=tf.nn.elu)
    x2 = layers.dropout(x2,keep_prob=0.7)
    logits = layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)
    return logits

