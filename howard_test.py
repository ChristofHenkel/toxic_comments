import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

max_features = 20000
maxlen = 150
embed_size = 50


train = pd.read_csv("assets/raw_data/train.csv")

train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("Nothing").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



Y_train = train[list_classes].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)


#def get_coefs(word,*arr):
#    return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('assets/embedding_models/glove/glove.twitter.27B.50d.txt'))
#
#all_embs = np.stack(embeddings_index.values())
#emb_mean,emb_std = all_embs.mean(), all_embs.std()
#emb_mean,emb_std


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            if embedding.shape[0] == 100:
                model[word] = embedding
        print("Done.",len(model)," words loaded!")
    return model

model = loadGloveModel('assets/embedding_models/glove/glove.twitter.27B.100d.txt')
words = list(model)
all_embs = np.asarray([model[word] for word in words[:1193513]])
emb_mean,emb_std = all_embs.mean(), all_embs.std()

del all_embs
#emb_mean,emb_std = 0.04399, 0.73192 #dim 50
#emb_mean,emb_std = 0.02631, 0.58371 #dim 100

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = model.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del model

graph = tf.Graph()
with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.int32, shape=(None,maxlen), name="input_x")
        y = tf.placeholder(tf.int32, shape=(None,6), name="input_y")
        keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")

    with tf.name_scope("Embedding"):
        #embedding = tf.get_variable("embedding", [len(tokenizer.word2index), 100], dtype=tf.float32,initializer=tf.constant_initializer(pre_embedding), trainable=False)
        embedding = tf.get_variable("embedding", [max_features, 50], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix))
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    with tf.variable_scope('forward'):
        stacked_fw_rnn = []
        for fw_Lyr in range(1):
            # fw_cell = tf.contrib.rnn.GRUCell(32)  # or True
            fw_cell = tf.contrib.rnn.BasicLSTMCell(50, forget_bias=1.0, state_is_tuple=True)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
            stacked_fw_rnn.append(fw_cell)
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        stacked_bw_rnn = []
        for bw_Lyr in range(1):
            # bw_cell = tf.contrib.rnn.GRUCell(32)  # or True
            bw_cell = tf.contrib.rnn.BasicLSTMCell(50, forget_bias=1.0, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
            stacked_bw_rnn.append(bw_cell)
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedded_input,dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    # outputs = tf.add(output_fw[:,:,-1],output_bw[:,:,-1])
    outputs = tf.reduce_mean(tf.stack([output_fw[:, :, -1], output_bw[:, :, -1]]), axis=0)

    x3 = layers.fully_connected(outputs, 50, activation_fn=tf.nn.relu)
    x3 = layers.dropout(x3, keep_prob=keep_prob)

    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        #tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
        #xent = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        #cost = tf.reduce_mean(xent, name='xent')
        cost = tf.losses.log_loss(predictions=logits, labels=y)

    #gradients = tf.gradients(cost, tf.trainable_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


epochs = 2
bsize = 512
train_iters = len(X_train) - bsize
steps = train_iters // bsize

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):

        step = 0
        while step * bsize < train_iters:
            batch_x = X_train[step*bsize:(step+1)*bsize]
            batch_y = Y_train[step*bsize:(step+1)*bsize]
            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:0.9})
            print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))

            step += 1

        #test_cost_ = sess.run(cost, feed_dict={x: X_valid[:2000],
        #                                       y: Y_valid[:2000],
        #                                       keep_prob:1})
        #print('valid loss: %s' %test_cost_)

    test = pd.read_csv("assets/raw_data/test.csv")
    list_sentences_test = test["comment_text"].fillna("Nothing").values
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

    num_batches = (len(X_test) // bsize) + 1


    res = np.zeros((len(X_test),6))
    for s in range(num_batches):
        print(s)
        batch_x_test = X_test[s * bsize:(s + 1) * bsize]
        logits_ = sess.run(logits, feed_dict={x: batch_x_test,
                                              keep_prob:1})
        res[s * bsize:(s + 1) * bsize] = logits_

    sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
    sample_submission[list_classes] = res
    sample_submission.to_csv("submissions/baseline4_howard_09.csv", index=False)