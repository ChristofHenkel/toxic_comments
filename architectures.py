import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from keras.layers import CuDNNGRU, Dropout, Bidirectional, BatchNormalization, SpatialDropout1D


class CNN:

    @staticmethod
    def text_cnn(embedding_matrix,x,keep_prob):
        """
        https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
        :param embedding_matrix:
        :param x:
        :param keep_prob:
        :return:
        """

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            #x2 = embedded_input
        pooled_outputs = []
        for i in range(3,6):
            x2 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=i, strides=1,activation=tf.nn.relu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=500-i, strides=1)
            pooled_outputs.append(x2)

        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)

        #fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def inception_1(embedding_matrix,x,keep_prob):
        """
        https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
        :param embedding_matrix:
        :param x:
        :param keep_prob:
        :return:
        """

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            #x2 = embedded_input
        pooled_outputs = []
        for i in range(3,6):
            x2 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=i, strides=1,activation=tf.nn.relu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=500-i, strides=1)
            x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
            pooled_outputs.append(x2)

        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)

        #fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def inception_2(embedding_matrix,x,keep_prob):
        """
        https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
        :param embedding_matrix:
        :param x:
        :param keep_prob:
        :return:
        """

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            #x2 = embedded_input
        pooled_outputs = []
        for i in [1,3,5]:
            x2 = tf.layers.conv1d(embedded_input, filters=64, kernel_size=i, strides=1,activation=tf.nn.relu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=500-i, strides=1)
            x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
            pooled_outputs.append(x2)

        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)

        #fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits


    def inception_3(self,embedding_matrix,x,keep_prob):
        """
        https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
        :param embedding_matrix:
        :param x:
        :param keep_prob:
        :return:
        """

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            #x2 = embedded_input
        pooled_outputs = []
        for i in [1,3,5]:
            x2 = tf.layers.conv1d(embedded_input, filters=64, kernel_size=i, strides=1,activation=self.prelu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=500-i, strides=1)
            x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
            pooled_outputs.append(x2)

        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)

        #fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def prelu(_x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg


    def inception_v3(self,embedding_matrix,x,keep_prob):
        """
        https://arxiv.org/pdf/1512.00567.pdf

        :return:
        """
        num_filters = 64

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            #x2 = embedded_input
        outputs = []

        x1 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1,activation=self.prelu)
        x1 = tf.layers.max_pooling1d(x1, pool_size=500-1, strides=1)
        x1 = tf.nn.dropout(x1, keep_prob=keep_prob)
        outputs.append(x1)

        x2 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1,activation=self.prelu)
        x2 = tf.layers.conv1d(x2, filters=num_filters, kernel_size=3, strides=1, activation=self.prelu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=500-3, strides=1)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        outputs.append(x2)

        x3 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1,activation=self.prelu)
        x3 = tf.layers.conv1d(x3, filters=num_filters, kernel_size=3, strides=1, activation=self.prelu)
        x3 = tf.layers.conv1d(x3, filters=num_filters, kernel_size=3, strides=2, activation=self.prelu)
        x3 = tf.layers.max_pooling1d(x3, pool_size=500-5, strides=1)
        x3 = tf.nn.dropout(x3, keep_prob=keep_prob)
        outputs.append(x3)

        h_pool = tf.concat(outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)

        #fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def inception_v4(embedding_matrix,x,keep_prob):
        """
        https://arxiv.org/pdf/1512.00567.pdf

        :return:
        """
        num_filters = 64

        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

            embedded_input = SpatialDropout1D(0.2)(embedded_input)
            # embedded_input = tf.transpose(embedded_input, [0, 2, 1])
            embedded_input = tf.cast(embedded_input, tf.float32)
        outputs = []

        x1 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1, activation=tf.nn.relu)
        x1 = tf.layers.max_pooling1d(x1, pool_size=500 - 1, strides=1)
        outputs.append(x1)

        x2 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1, activation=tf.nn.relu)
        x2 = tf.layers.conv1d(x2, filters=num_filters, kernel_size=3, strides=1, activation=tf.nn.relu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=500 - 3, strides=1)

        outputs.append(x2)

        x3 = tf.layers.conv1d(embedded_input, filters=num_filters, kernel_size=1, strides=1, activation=tf.nn.relu)
        x3 = tf.layers.conv1d(x3, filters=num_filters, kernel_size=3, strides=1, activation=tf.nn.relu)
        x3 = tf.layers.conv1d(x3, filters=num_filters, kernel_size=3, strides=1, activation=tf.nn.relu)
        x3 = tf.layers.max_pooling1d(x3, pool_size=500 - 5, strides=1)

        outputs.append(x3)

        h_pool = tf.concat(outputs, 2)
        h_pool_flat = tf.layers.flatten(h_pool)
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob=keep_prob)

        # fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 64)
        logits = tf.contrib.layers.fully_connected(h_pool_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def vgg_4(embedding_matrix,x,keep_prob):

        depth = 4

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)
        fc1 = tf.contrib.layers.fully_connected(conv_output, 64)
        logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def vgg_5(embedding_matrix,x,keep_prob):

        depth = 5

        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)
        fc1 = tf.contrib.layers.fully_connected(conv_output, 64)
        logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def vgg_5b(embedding_matrix,x,keep_prob):

        depth = 5

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)

        logits = tf.contrib.layers.fully_connected(conv_output, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def vgg_5c(embedding_matrix,x,keep_prob):

        depth = 5

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.concat([tf.reduce_max(x2, axis=1),tf.reduce_mean(x2, axis=1)],axis=1)

        logits = tf.contrib.layers.fully_connected(conv_output, 6, activation_fn=tf.nn.sigmoid)
        return logits

    def vgg_5_dilations(self,embedding_matrix,x,keep_prob):
        depth = 5
        #dilation_rates = [0, 2, 4, 8, 16]
        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=self.prelu)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=self.prelu)
            #x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)
        #means = tf.reduce_mean(x2, axis=1)
        #conv_output = tf.concat([conv_output,means])

        logits = tf.contrib.layers.fully_connected(conv_output, 6, activation_fn=tf.nn.sigmoid)
        return logits



    @staticmethod
    def vgg_6(embedding_matrix,x,keep_prob):

        depth = 6

        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)
        fc1 = tf.contrib.layers.fully_connected(conv_output, 64)
        logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def vgg_5_elu(embedding_matrix, x, keep_prob):
        depth = 5

        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]],
                                        dtype=tf.float32, initializer=tf.constant_initializer(embedding_matrix),
                                        trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
            x2 = embedded_input

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)
        fc1 = tf.contrib.layers.fully_connected(conv_output, 64, activation_fn=tf.nn.elu)
        logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def cnn_v1(x, vocab_size, nb_filter, filter_kernels, dense_outputs):
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

        x2 = tf.layers.conv1d(embedded_input, filters=nb_filter, kernel_size=filter_kernels[0])
        x2 = tf.layers.max_pooling1d(x2, 3, 1)

        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[1], )
        x2 = tf.layers.max_pooling1d(x2, 3, 1)

        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[2])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[3])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[4])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[5])

        x2 = tf.layers.max_pooling1d(x2, 3, 1)
        x2 = tf.layers.flatten(x2)

        x2 = layers.fully_connected(x2, dense_outputs)
        x2 = layers.dropout(x2, keep_prob=0.5)
        x2 = layers.fully_connected(x2, dense_outputs)
        x2 = layers.dropout(x2, keep_prob=0.5)
        logits = layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def cnn_rnn_v1(x, vocab_size, nb_filter, filter_kernels, dense_outputs):
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

        x2 = tf.layers.conv1d(embedded_input, filters=nb_filter, kernel_size=filter_kernels[0])
        x2 = tf.layers.max_pooling1d(x2, 3, 1)

        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[1])
        x2 = tf.layers.max_pooling1d(x2, 3, 1)

        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[2])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[3])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[4])
        x2 = tf.layers.conv1d(x2, filters=nb_filter, kernel_size=filter_kernels[5])

        x2 = tf.layers.max_pooling1d(x2, 3, 1)
        # x2 = tf.layers.flatten(x2)

        fw_cell = tf.contrib.rnn.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(64, forget_bias=1.0, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x2, dtype=tf.float32)
        output_fw, output_bw = outputs

        outputs = tf.add(output_fw, output_bw)
        outputs = tf.contrib.layers.flatten(outputs)

        x2 = layers.fully_connected(outputs, 256, activation_fn=tf.nn.elu)
        x2 = layers.fully_connected(x2, 64, activation_fn=tf.nn.elu)
        x2 = layers.dropout(x2, keep_prob=0.7)
        logits = layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)
        return logits


class CCNN:

    @staticmethod
    def vgg(x,depth, char_vocab_size, is_training, embed_dim= 300):

        embedding = tf.get_variable("embedding", [char_vocab_size, embed_dim], dtype=tf.float32)
        x2 = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)

            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)

        fc1 = tf.contrib.layers.fully_connected(conv_output, 32, activation_fn=tf.nn.elu)

        return fc1

    @staticmethod
    def vgg_bn(x,depth, char_vocab_size, is_training, embed_dim= 300):

        embedding = tf.get_variable("embedding", [char_vocab_size, embed_dim], dtype=tf.float32)
        x2 = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        for i in range(3, 3 + depth):
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.batch_normalization(inputs=x2, axis=1,momentum=0.9, training=is_training, fused=True)
            x2 = tf.nn.relu(x2)
            x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1)
            x2 = tf.layers.batch_normalization(inputs=x2, axis=1,momentum=0.9, training=is_training, fused=True)
            x2 = tf.nn.relu(x2)
            x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

        conv_output = tf.reduce_max(x2, axis=1)

        fc1 = tf.contrib.layers.fully_connected(conv_output, 64, activation_fn=tf.nn.elu)
        logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)

        return logits


class CRNN:

    @staticmethod
    def cnn_rnn(embedding_matrix,x,keep_prob):
        embedding = tf.get_variable("embedding", [embedding_matrix.get_shape()[0],embedding_matrix.get_shape()[1]], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        x2 = embedded_input
        #for i in range(3, 3 + 2):
        #    x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
        #    x2 = tf.layers.conv1d(x2, filters=2 ** i, kernel_size=3, strides=1, activation=tf.nn.elu)
        #    x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

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

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #with tf.variable_scope('fc'):
        #    prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

        logits = tf.contrib.layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def cudnn_rnn(embedding_matrix,x,keep_prob):
        embedding = tf.get_variable("embedding", [embedding_matrix.get_shape()[0],embedding_matrix.get_shape()[1]], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
        #x2 = Dropout(1-keep_prob)(x2)

        outputs = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #with tf.variable_scope('fc'):
        #    prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

        logits = tf.contrib.layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def cudnn_cnn_rnn(embedding_matrix,x,keep_prob):
        embedding = tf.get_variable("embedding", [embedding_matrix.get_shape()[0],embedding_matrix.get_shape()[1]], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        x2 = tf.layers.conv1d(embedded_input, filters=256, kernel_size=3, strides=1)
        x2 = tf.layers.conv1d(x2, filters=256, kernel_size=3, strides=1)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2,keep_prob=keep_prob)
        outputs = BatchNormalization()(x2)

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #with tf.variable_scope('fc'):
        #    prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

        logits = tf.contrib.layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def cudnn_cnn_rnn2(embedding_matrix, x, keep_prob):
        embedding = tf.get_variable("embedding", [embedding_matrix.get_shape()[0], embedding_matrix.get_shape()[1]],
                                    dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
        x2 = embedded_input
        for i in range(3, 3 + 2):
            x2 = tf.layers.conv1d(x2, filters=2 ** (i+4), kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=tf.nn.relu)
            x2 = tf.layers.conv1d(x2, filters=2 ** (i+4), kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=tf.nn.relu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        outputs = BatchNormalization()(x2)

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        # with tf.variable_scope('fc'):
        #    prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

        logits = tf.contrib.layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

class BIRNN:

    @staticmethod
    def _attention_mechanism(outputs, attention_layer_size, rnn_units,maxSeqLength):
        """
        Attention Network to average the output of the bidirectional RNN network.
        Small neural network, which should learn the 'importance' of an individual word for classification.
        :param outputs: The outut of the Rnn network.
        :param attention_layer_size: Size of the hidden layer.
        :return: alpha-coefficients which produce an weighted average of the rnn_outputs.
        """
        # Reshape outputs to tensor of shape (batch_size * self.maxSeqLength, 2 * self.lstm_size)
        # So each output from one Rnn cell is fed into the connected layer once in a time.
        outputs = tf.reshape(outputs, [-1, 2 * rnn_units])
        hidden_layer = layers.fully_connected(outputs, attention_layer_size, activation_fn=tf.nn.relu)
        attention_logits = layers.fully_connected(hidden_layer, 1, activation_fn=None)

        #Reshape attention_logits, such that is of the shape (batch_size, self.maxSeqLength, 1).
        attention_logits = tf.reshape(attention_logits, [-1, maxSeqLength, 1])

        # Apply softmax to the maxSeqLength dimensions, i.e. the different outputs.
        # alphas has shape (batch_size, self.maxSeqLength, 1)
        alphas = tf.nn.softmax(attention_logits, dim=1)

        # TODO: Idea for nomalisation of the softmax.
        # unnormed_softmax = tf.exp(attention_logits)
        # softmax_norm = tf.reduce_sum(unnormed_softmax * batch_masking, axis=1)
        # alphas = unnormed_softmax / softmax_norm

        # Store alphas as class variable for visualization of the attentions.


        return alphas


    @staticmethod
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

        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def pavel_all_outs(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

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

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def ugrnn_all_outs(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        with tf.variable_scope('forward'):

            fw_cell1 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
            fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)
            fw_cell2 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
            stacked_fw_rnn = [fw_cell1,fw_cell2]
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

        with tf.variable_scope('backward'):
            bw_cell1 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
            bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)
            bw_cell2 = tf.contrib.rnn.UGRNNCell(64,activation=tf.nn.elu)
            stacked_bw_rnn = [bw_cell1,bw_cell2]
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
        output_fw, output_bw = outputs

        outputs = tf.concat([output_fw, output_bw], axis = 2)

        outputs = tf.transpose(outputs, [0, 2, 1])

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def pavel_all_outs_BN(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2,keep_prob=keep_prob)
        outputs = BatchNormalization()(x2)


        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def pavel_all_outs_BNb(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2,keep_prob=keep_prob)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        outputs = BatchNormalization()(x2)


        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def pavel_all_outs_BNd(embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

        x2 = SpatialDropout1D(0.2)(embedded_input)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2,keep_prob=keep_prob)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        outputs = BatchNormalization()(x2)


        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        #x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    def pavel_attention(self,embedding_matrix, x, keep_prob):

        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

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

        #outputs = tf.transpose(outputs, [0, 2, 1])

        alphas = self._attention_mechanism(outputs, attention_layer_size=10,rnn_units=64,maxSeqLength=500)
        # encodings is of shape (batch_size, 2 * self.lstm_size)
        encodings = tf.reduce_sum(alphas * outputs, 1)

        x3 = layers.fully_connected(encodings, 32, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits

    @staticmethod
    def ugrnn(embedding_matrix, x, keep_prob):
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def rnn_cnn(embedding_matrix,x,keep_prob,z):


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





        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
        outputs_with_cnn = tf.concat(x3, z)
        logits = layers.fully_connected(outputs_with_cnn, 6, activation_fn=tf.nn.sigmoid)
        return logits


class CCAPS:

    def __init__(self):
        pass


    def routing(self,input, b_IJ, iter_routing=3):
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
                if r_iter == iter_routing - 1:
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    assert s_J.get_shape() == [bsize, 1, 10, 16, 1]

                    # line 6:
                    # squash using Eq.1,
                    v_J = self.squash(s_J)
                    assert v_J.get_shape() == [bsize, 1, 10, 16, 1]
                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    v_J_tiled = tf.tile(v_J, [1, num_caps, 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                    assert u_produce_v.get_shape() == [bsize, num_caps, 10, 1, 1]

                    # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                    b_IJ += u_produce_v

        return (v_J)


    @staticmethod
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

    def single_conv_caps(self,x, char_vocab_size, bsize = 128, embed_dim = 50):


        with tf.variable_scope('Embedding'):
            embedding = tf.get_variable("embedding", [char_vocab_size, embed_dim], dtype=tf.float32)
            x2 = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv1d(x2, filters=128, kernel_size=50, strides=2)

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(conv1, filters=32 * 8,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 234 * 32, 8, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 234 * 32, 1, 8, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)

            logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits


class CAPS:

    def __init__(self):
        pass

    @staticmethod
    def prelu(_x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def routing(self,input, b_IJ, iter_routing=3, caps_dim_in=6, caps_dim_out=8, num_caps_out=6):
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
                    v_J = self.squash(s_J)
                    assert v_J.get_shape() == [bsize, 1, num_caps_out, caps_dim_out, 1]
                elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)

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

    @staticmethod
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

    def single_conv_caps(self,embedding_matrix, x, keep_prob, bsize):


        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")


        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=50, strides=1)

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(conv1, filters=32 * 8,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * 32, 8, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, 8, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)

            logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits

    def single_conv_caps_b(self,embedding_matrix, x, keep_prob, bsize):


        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")


        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=50, strides=1,activation=self.prelu)

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(conv1, filters=32 * 8,
                                        kernel_size=9,
                                        strides=2,
                                        activation=self.prelu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * 32, 8, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, 8, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)

            logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits


    def caps_4(self,embedding_matrix, x, keep_prob, bsize):

        n_caps = 8
        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")


        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=50, strides=1)

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(conv1, filters=32 * 8,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * 32, 8, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, 8, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)

            logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits

    def inception_caps(self,em, x, keep_prob, bsize):
        n_caps = 6
        n_capfilter = 32
        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(em, x, name="embedded_input")

        conv1 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=1, strides=2)
        conv3 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=3, strides=2)
        conv5 = tf.layers.conv1d(embedded_input, filters=128, kernel_size=5, strides=2)
        conv = tf.concat([conv1, conv3, conv5], axis=1)
        conv = tf.transpose(conv,[0,2,1])


        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(conv, filters=n_capfilter * n_caps,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * n_capfilter, n_caps, 1))
            capsules = self.squash(capsules)

        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, n_caps, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            # about the reason of using 'batch_size', see issue #21
            b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 6, 1, 1], dtype=np.float32))
            capsules = self.routing(input, b_IJ, num_caps_out=6, caps_dim_out=8)
            capsules = tf.squeeze(capsules, axis=1)

        #cap_norms = tf.norm(capsules, axis=2)[:,:,0]
        #cap_norms = tf.minimum(tf.maximum(cap_norms,0),1)

        flat_capsules = tf.layers.flatten(capsules)

        logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

        return logits


    def rnn_caps(self,embedding_matrix, x, keep_prob, bsize):

        n_caps = 8
        n_capfilter = 32
        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")


        with tf.variable_scope('Gru_layer'):
            with tf.variable_scope('fw'):
                fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)

            with tf.variable_scope('bw'):
                bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell1, bw_cell1, embedded_input, dtype=tf.float32)
            output_fw, output_bw = outputs

            outputs = tf.concat([output_fw, output_bw], axis=2)

            outputs = tf.transpose(outputs, [0, 2, 1])

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(outputs, filters=n_capfilter * n_caps,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * n_capfilter, n_caps, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, n_caps, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)

            logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits


    def rnn_caps2(self,embedding_matrix, x, keep_prob, bsize):

        n_caps = 8
        n_capfilter = 32
        with tf.name_scope("Embedding"):
            #embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            #embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")


        with tf.variable_scope('Gru_layer'):
            with tf.variable_scope('fw'):
                fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)

            with tf.variable_scope('bw'):
                bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell1, bw_cell1, embedded_input, dtype=tf.float32)
            output_fw, output_bw = outputs

            outputs = tf.concat([output_fw, output_bw], axis=2)

            outputs = tf.transpose(outputs, [0, 2, 1])

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(outputs, filters=n_capfilter * n_caps,
                                        kernel_size=3,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 63 * n_capfilter, n_caps, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 63 * n_capfilter, 1, n_caps, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)
            dropped_flat_capsules = tf.nn.dropout(flat_capsules, keep_prob)

            logits = tf.contrib.layers.fully_connected(dropped_flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits

    def rnn_caps3(self,embedding_matrix, x, keep_prob, bsize):

        n_caps = 8
        n_capfilter = 32
        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")


        with tf.variable_scope('Gru_layer'):
            with tf.variable_scope('fw'):
                fw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob)

            with tf.variable_scope('bw'):
                bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
                bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell1, bw_cell1, embedded_input, dtype=tf.float32)
            output_fw, output_bw = outputs

            outputs = tf.concat([output_fw, output_bw], axis=2)

            outputs = tf.transpose(outputs, [0, 2, 1])

        # Primary Capsules layer, return [batch_size, ? , 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(outputs, filters=n_capfilter * n_caps,
                                        kernel_size=3,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 63 * n_capfilter, n_caps, 1))
            capsules = self.squash(capsules)


        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 63 * n_capfilter, 1, n_caps, 1))

            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 10, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ)
                capsules = tf.squeeze(capsules, axis=1)

            flat_capsules = tf.layers.flatten(capsules)
            dropped_flat_capsules = tf.nn.dropout(flat_capsules, keep_prob)

            logits = tf.contrib.layers.fully_connected(dropped_flat_capsules, 6, activation_fn=tf.nn.sigmoid)

            return logits

    def cudnnrnn_caps(self,em,x,keep_prob,bsize):

        n_caps = 6
        n_capfilter = 32
        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(em, x, name="embedded_input")

        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)

        outputs = tf.transpose(x2, [0, 2, 1])
        with tf.variable_scope('PrimaryCaps_layer'):
            capsules = tf.layers.conv1d(outputs, filters=n_capfilter * n_caps,
                                        kernel_size=9,
                                        strides=2,
                                        activation=tf.nn.relu)
            capsules = tf.expand_dims(capsules, 3)
            capsules = tf.reshape(capsules, (bsize, 60 * n_capfilter, n_caps, 1))
            capsules = self.squash(capsules)

        with tf.variable_scope('FCCaps_layer'):
            # digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            # caps2 = digitCaps(capsules_squashed)

            input = tf.reshape(capsules, shape=(bsize, 60 * 32, 1, n_caps, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            # about the reason of using 'batch_size', see issue #21
            b_IJ = tf.constant(np.zeros([bsize, input.shape[1].value, 6, 1, 1], dtype=np.float32))
            capsules = self.routing(input, b_IJ, num_caps_out=6, caps_dim_out=8)
            capsules = tf.squeeze(capsules, axis=1)

        # cap_norms = tf.norm(capsules, axis=2)[:,:,0]
        # cap_norms = tf.minimum(tf.maximum(cap_norms,0),1)

        flat_capsules = tf.layers.flatten(capsules)

        logits = tf.contrib.layers.fully_connected(flat_capsules, 6, activation_fn=tf.nn.sigmoid)
        return logits


class DENSE:

    @staticmethod
    def rectangle_512x3(embedding_matrix,x,keep_prob):
        with tf.name_scope("Embedding"):
            # embedding = tf.get_variable("embedding", tf.shape(embedding_matrix), dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding_matrix, x, name="embedded_input")

            #embedded_input = SpatialDropout1D(0.2)(embedded_input)
            #embedded_input = tf.cast(embedded_input, tf.float32)

        h1 = layers.fully_connected(embedded_input, 512, activation_fn=tf.nn.relu)
        h2 = layers.fully_connected(h1, 512, activation_fn=tf.nn.relu)
        h3 = layers.fully_connected(h2, 512, activation_fn=tf.nn.relu)
        h_flat = tf.layers.flatten(h3)
        h_flat = tf.nn.dropout(h_flat, keep_prob=keep_prob)

        logits = layers.fully_connected(h_flat, 6, activation_fn=tf.nn.sigmoid)
        return logits

class CNNRNN:

    @staticmethod
    def cudnn_cnn_rnn2(embedding_matrix, x, keep_prob):
        embedding = tf.get_variable("embedding", [embedding_matrix.get_shape()[0], embedding_matrix.get_shape()[1]],
                                    dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
        x2 = embedded_input
        for i in range(3, 3 + 2):
            x2 = tf.layers.conv1d(x2, filters=2 ** (i+4), kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=tf.nn.relu)
            x2 = tf.layers.conv1d(x2, filters=2 ** (i+4), kernel_size=3, strides=1,dilation_rate=2**(i-3),activation=tf.nn.relu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)
        x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x2)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)
        outputs = BatchNormalization()(x2)

        maxs = tf.reduce_max(outputs, axis=2)
        means = tf.reduce_mean(outputs, axis=2)
        last = outputs[:, :, -1]
        x3 = tf.concat([maxs, means, last], axis=1)

        # with tf.variable_scope('fc'):
        #    prelogits = layers.fully_connected(outputs, 32, activation_fn=tf.nn.elu)

        logits = tf.contrib.layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        return logits