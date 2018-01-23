import numpy as np
import pandas as pd

from preprocess_utils import Tokenizer
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.preprocessing import sequence
import sklearn.model_selection as sk


train = pd.read_csv("assets/raw_data/train.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sentences_train = train["comment_text"].fillna("Nothing").values
labels = train[list_classes].values

sentences_train, sentences_valid, Y_train, Y_valid = sk.train_test_split(sentences_train, labels, test_size=0.1, random_state=42)

#test = pd.read_csv("assets/raw_data/test.csv")
#list_sentences_test = test["comment_text"].fillna("Nothing").values

tokenizer = Tokenizer(min_count_words=5,min_count_chars=200)
#tokenizer.fit_on_texts(sentences_train)

def load_train_data(list_sentences):
    list_tokenized = tokenizer.texts_to_sequences(list_sentences)
    X_train = sequence.pad_sequences(list_tokenized, maxlen=150)
    return X_train

def extract_gazetter_from_text(text):
    gazetter = [1 if w in text else 0 for w in tokenizer.bad_words]
    return gazetter

def extract_gazetter_from_seq(X_train):
    print('getting gazetter features')
    G_train = [tokenizer.seq2gazetter(seq) for seq in X_train]
    return G_train

#X_train = load_train_data(sentences_train)
#X_valid = load_train_data(sentences_valid)
#G_train = extract_gazetter_from_seq(X_train)
#G_valid = extract_gazetter_from_seq(X_valid)

G_train = [extract_gazetter_from_text(text) for text in sentences_train]
G_valid = [extract_gazetter_from_text(text) for text in sentences_valid]

epochs = 4
bsize = 512
train_iters = len(G_train) - bsize
steps = train_iters // bsize

graph = tf.Graph()

with graph.as_default():

    tf.set_random_seed(1)
    x = tf.placeholder(tf.float32, shape=(None,len(G_train[0])), name="input_x")
    y = tf.placeholder(tf.int32, shape=(None,6), name="input_y")
    x2 = layers.fully_connected(x, 20, activation_fn=tf.nn.tanh)
    x3 = layers.dropout(x2, keep_prob=0.8)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
    cost = tf.losses.log_loss(predictions=logits, labels=y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)



with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        costs = []
        step = 0
        while step * bsize < train_iters:
            batch_x = G_train[step*bsize:(step+1)*bsize]
            batch_y = Y_train[step*bsize:(step+1)*bsize]

            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y})
            print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))

            costs.append(cost_)
            step += 1

        valid_cost_ = sess.run(cost, feed_dict={x: G_valid,
                                                y: Y_valid})

        avg_cost = np.log(np.mean(np.exp(costs)))
        print('avg train loss: %s' % avg_cost)
        print('valid loss: %s' % valid_cost_)
