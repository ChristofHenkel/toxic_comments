
from cnn_architectures import cnn_rnn_v1 as baseline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from utils import create_embedding_matrix
# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.contrib import layers
from preprocess_utils import CNNTransformer

maxlen = 1000


train = pd.read_csv("assets/raw_data/train.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("Nothing").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_train = train[list_classes].values



transformer = CNNTransformer()
transformer.max_len = maxlen
transformer.create_vocabulary(list_sentences_train)

split_at = int(0.9*len(list_sentences_train))


Y_test = Y_train[split_at:]
Y_train = Y_train[:split_at]

#X_train = np.zeros((len(list_sentences_train),maxlen,transformer.vocab_size))
#for k, text in enumerate(list_sentences_train):
#    if k % 100 == 0:
#        print(k)
#    matrix = transformer.convert_text_to_matrix(text)
#    X_train[k] = matrix

X_train2 = np.zeros((len(list_sentences_train),maxlen))
for k, text in enumerate(list_sentences_train):
    if k % 100 == 0:
        print(k)
    seq = transformer.convert_text_to_seq(text)
    X_train2[k] = seq

X_test2 = X_train2[split_at:]
X_train2 = X_train2[:split_at]

# Model params

#nb_filter = 256     # Filters for conv layers
#dense_outputs = 1024        # Number of units in the dense layer

nb_filter = 64
dense_outputs = 256
filter_kernels = [7, 7, 3, 3, 3, 3]     # Conv layer kernel size

# Compile/fit params
batch_size = 256
nb_epoch = 10

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(1)

    #x = tf.placeholder(dtype=tf.int32,shape=(None,maxlen,transformer.vocab_size))
    x = tf.placeholder(dtype=tf.int32, shape=(None, maxlen))
    y = tf.placeholder(dtype=tf.int32,shape=(None,6))

    logits = baseline(x,transformer.vocab_size,nb_filter,filter_kernels,dense_outputs)

    loss = tf.losses.log_loss(predictions=logits, labels=y)

    gradients = tf.gradients(loss, tf.trainable_variables())
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for e in range(10):
        s = 0
        while (s+1)*batch_size < len(X_train2):
            batch_x = X_train2[s*batch_size:(s+1)*batch_size]
            batch_y = Y_train[s*batch_size:(s+1)*batch_size]
            loss_, _ = sess.run([loss,optimizer],feed_dict={x:batch_x,y:batch_y})
            print('e %s -- s %s -- log-loss %s' %(e,s,loss_))
            s += 1
        loss_, _ = sess.run([loss, optimizer], feed_dict={x: X_test2, y: Y_test})
        print('Validation e %s -- log-loss %s' % (e, loss_))

