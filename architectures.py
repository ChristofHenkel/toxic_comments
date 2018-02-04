import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

class CNN:

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


class BIRNN:

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
        #outputs = outputs[:,:,-1]

        x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
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

    def single_conv_caps(self,embedding_matrix, x, keep_prob, bsize):


        with tf.name_scope("Embedding"):
            embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
            embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")


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


    def rnn_caps(self,embedding_matrix, x, keep_prob, bsize):


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
            capsules = tf.layers.conv1d(outputs, filters=32 * 8,
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