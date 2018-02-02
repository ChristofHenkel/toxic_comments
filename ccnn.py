import pandas as pd
from preprocess_utils import Preprocessor
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import numpy as np
import os

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
maxlen = 2000
DEPTH = 7
EPOCHS = 15
model_name = 'vgg7_2'
root = '/home/christof/kaggle/toxic_comments/'
fp = 'models/CCNN/' + model_name + '/'
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
    x = tf.placeholder(dtype=tf.int32,shape=(None,maxlen))
    y = tf.placeholder(dtype=tf.float32,shape=(None,6))
    keep_prob = tf.placeholder(dtype=tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    embedding = tf.get_variable("embedding", [preprocessor.char_vocab_size, 300], dtype=tf.float32)
    x2 = tf.nn.embedding_lookup(embedding, x, name="embedded_input")


    for i in range(3,3+DEPTH):
        x2 = tf.layers.conv1d(x2, filters=2**i, kernel_size=3, strides=1, activation=tf.nn.elu)
        x2 = tf.layers.conv1d(x2, filters=2**i, kernel_size=3, strides=1, activation=tf.nn.elu)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

    conv_output = tf.reduce_max(x2, axis=1)

    fc1 = tf.contrib.layers.fully_connected(conv_output, 64, activation_fn=tf.nn.elu)
    logits = tf.contrib.layers.fully_connected(fc1, 6, activation_fn=tf.nn.sigmoid)

    loss = binary_crossentropy(y, logits)
    cost = tf.losses.log_loss(labels=y,predictions=logits)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)
    (_, auc_update_op) = tf.contrib.metrics.streaming_auc(
        predictions=logits,
        labels=y,
        curve='ROC')
    saver = tf.train.Saver()

def save(saver, sess, epoch):
    print('saving model...', end='')
    model_name = 'e%s.ckpt' %epoch
    s_path = saver.save(sess, logs_path + model_name)
    print("Model saved in file: %s" % s_path)

bsize = 512
train_iters = len(X_train) - bsize
valid_iters = len(X_valid) - bsize
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(EPOCHS):
        step = 0
        tf.local_variables_initializer().run(session=sess)
        conv_out_train = np.zeros((len(X_train),512))
        while step * bsize < train_iters:
            batch_x = X_train[step * bsize:(step + 1) * bsize]
            batch_y = Y_train[step * bsize:(step + 1) * bsize]
            cost_ , _, roc_auc_train,conv_out_train_batch = sess.run([cost,optimizer,auc_update_op,conv_output],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:0.7,
                                                             is_training:True})

            print('e%s -- s%s -- cost: %s' %(epoch,step,cost_))
            conv_out_train[step * bsize:(step + 1) * bsize] = conv_out_train_batch
            step +=1

        vstep = 0
        vcosts = []
        tf.local_variables_initializer().run(session=sess)
        conv_out_valid = np.zeros((len(X_valid), 512))
        while vstep * bsize < valid_iters:
            test_cost_, roc_auc_test,conv_out_valid_batch = sess.run([cost,auc_update_op,conv_output], feed_dict={x: X_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   y: Y_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   keep_prob: 1,
                                                   is_training: False
                                                   })

            vcosts.append(test_cost_)
            conv_out_valid[vstep * bsize:(vstep + 1) * bsize] = conv_out_valid_batch
            vstep += 1
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        print('valid loss: %s' % avg_cost)
        print('roc auc test : {:.4}'.format(roc_auc_test))
        print('roc auc train : {:.4}'.format(roc_auc_train))
        results.loc[len(results)] = [epoch, roc_auc_test, roc_auc_train]
        results.to_csv(fp + 'results.csv')
        save(saver, sess, epoch)
        np.save('ccnn_' + str(epoch),np.concatenate((conv_out_train,conv_out_valid)))