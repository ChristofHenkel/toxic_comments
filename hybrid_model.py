
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from preprocess_utils import Tokenizer
from utils import create_embedding_matrix,load_glove_embedding
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.keras.api.keras.preprocessing import text, sequence
#from keras.callbacks import EarlyStopping, ModelCheckpoint


max_features = 38000
maxlen = 150


train = pd.read_csv("assets/raw_data/train.csv")
test = pd.read_csv("assets/raw_data/test.csv")
train = train.sample(frac=1,random_state=1)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


list_sentences_train = train["comment_text"].fillna("Nothing").values
Y_train = train[list_classes].values

split_at = int(len(list_sentences_train)*0.9)
list_sentences_valid = list_sentences_train[split_at:]
list_sentences_train = list_sentences_train[:split_at]
Y_valid = Y_train[split_at:]
Y_train = Y_train[:split_at]

list_sentences_test = test["comment_text"].fillna("Nothing").values



#tokenizer = text.Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(list(np.concatenate((list_sentences_train,list_sentences_test))))

tokenizer = Tokenizer(min_count_words=5,min_count_chars=200)
tokenizer.fit_on_texts(list_sentences_train)

def load_train_data(list_sentences):
    list_tokenized = tokenizer.texts_to_sequences(list_sentences)
    X_train = sequence.pad_sequences(list_tokenized, maxlen=maxlen)
    return X_train

def extract_gazetter(X_train):
    print('getting gazetter features')
    G_train = [tokenizer.seq2gazetter(seq) for seq in X_train]
    return G_train

def extract_charseq(X_train):

    Z_train = tokenizer.seqs_to_char_sequences(X_train, maxlenseq=12)
    return Z_train


X_train = load_train_data(list_sentences_train)
X_valid = load_train_data(list_sentences_valid)
Z_train = extract_charseq(X_train)
Z_valid = extract_charseq(X_valid)

def load_test_data():

    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
    return X_test



#need to check if with X_test its better
#pre_embedding = create_embedding_matrix(np.concatenate((X_train,X_test)),
#                                        tokenizer.word2index,
#                                        mode= 'glove')
pre_embedding = load_glove_embedding(tokenizer.word2index)

bsize = 512

train_iters = len(X_train) - bsize
vbsize = 512
valid_iters = len(X_valid) - vbsize
epochs = 3
steps = train_iters // bsize

def run_gazetter_model():
    G_train = extract_gazetter(X_train)
    G_valid = extract_gazetter(X_valid)
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

                cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,y:batch_y})
                print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))

                costs.append(cost_)
                step += 1

            vstep = 0
            vcosts = []
            while step * bsize <valid_iters:
                valid_cost_ = sess.run(cost, feed_dict={x: G_valid[:2000],
                                                       y: Y_valid[:2000],
                                                       })
                vcosts.append(valid_cost_)
                vstep +=1
            avg_cost = costs[-(valid_iters // bsize):]
            print('avg train loss: %s' % avg_cost)
            print('valid loss: %s' % valid_cost_)


G_train = extract_gazetter(X_train)
G_valid = extract_gazetter(X_valid)

graph = tf.Graph()
with graph.as_default():
    # tf Graph input

    with tf.name_scope("Input"):
        x = tf.placeholder(tf.int32, shape=(None,maxlen), name="input_x")
        z = tf.placeholder(tf.int32, shape=(None,maxlen,12), name="input_z")
        g = tf.placeholder(tf.float32, shape=(None, len(G_train[0])), name="input_g")
        y = tf.placeholder(tf.int32, shape=(None,6), name="input_y")
        keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")
        keep_prob_rnn = tf.placeholder(dtype=tf.float32, name="input_keep_prob_rnn")

    embedding2 = tf.get_variable("embeddingchar", [len(tokenizer.char2index), 21], dtype=tf.float32)
    embedded_input2 = tf.nn.embedding_lookup(embedding2, z, name="embedded_input")

    z2 = tf.layers.conv2d(embedded_input2,filters=32,kernel_size=3,padding='SAME')
    z2 = tf.layers.max_pooling2d(z2,(1,3),1)
    z2 = layers.dropout(z2, keep_prob=keep_prob)
    z2 = tf.layers.conv2d(z2,filters=32,kernel_size=3,padding='SAME')
    z2 = layers.dropout(z2, keep_prob=keep_prob)
    z2 = tf.layers.max_pooling2d(z2,(1,3),1)
    z2 = tf.layers.conv2d(z2, filters=50, kernel_size=1, strides=(1,8), padding='SAME')
    z2 = z2[:,:,0,:]


    with tf.name_scope("Embedding"):
        #embedding = tf.get_variable("embedding", [len(tokenizer.word2index), 100], dtype=tf.float32,initializer=tf.constant_initializer(pre_embedding), trainable=False)
        embedding = tf.get_variable("embedding", [len(tokenizer.word2index), 100], dtype=tf.float32)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
        # Creates Tensor with TensorShape([Dimension(batchsize), Dimension(nsteps), Dimension(embed_size)])
        embedded_input_both = tf.concat([embedded_input,z2],2)
        #embedded_input_both = z2
    with tf.name_scope("RNN"):

        #fw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
        #fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.8)
        #bw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
        #bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.8)

        with tf.variable_scope('forward'):
            stacked_fw_rnn = []
            for fw_Lyr in range(1):
                #fw_cell = tf.contrib.rnn.GRUCell(32)  # or True
                fw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob_rnn)
                stacked_fw_rnn.append(fw_cell)
            fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

        with tf.variable_scope('backward'):
            stacked_bw_rnn = []
            for bw_Lyr in range(1):
                #bw_cell = tf.contrib.rnn.GRUCell(32)  # or True
                bw_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=keep_prob_rnn)
                stacked_bw_rnn.append(bw_cell)
            bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedded_input,dtype=tf.float32)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input_both, dtype=tf.float32)
        output_fw, output_bw = outputs

        #outputs = tf.add(output_fw[:,:,-1],output_bw[:,:,-1])
        outputs = tf.reduce_mean(tf.stack([output_fw[:,:,-1],output_bw[:,:,-1]]),axis=0)

        mean_embedding = tf.reduce_mean(embedded_input,axis=1)
        x2 = tf.concat([outputs, mean_embedding],axis=1)
        x2 = tf.nn.l2_normalize(x2, 1, epsilon=1e-12, name=None)

        #x2 = outputs
        #x3 = layers.fully_connected(x2, 512, activation_fn=tf.nn.elu)
        #x3 = layers.dropout(x3, keep_prob=keep_prob)
        x3 = layers.fully_connected(x2, 32, activation_fn=tf.nn.elu)
        x3 = layers.dropout(x3, keep_prob=keep_prob)

        g2 = layers.fully_connected(g, 20, activation_fn=tf.nn.tanh)
        g2 = layers.dropout(g2, keep_prob=keep_prob)

        x3 = tf.concat([x3,g2],axis=1)


        #x3 = layers.fully_connected(x3, 1024, activation_fn=tf.nn.elu)
        #x3 = layers.dropout(x3, keep_prob=keep_prob)

        logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)
        #tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
        #xent = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        #cost = tf.reduce_mean(xent, name='xent')
        cost = tf.losses.log_loss(predictions=logits, labels=y)

    #gradients = tf.gradients(cost, tf.trainable_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session(graph=graph) as sess:
    tf.set_random_seed(1)
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):

        step = 0
        costs = []
        while step * bsize < train_iters:
            batch_x = X_train[step*bsize:(step+1)*bsize]
            batch_y = Y_train[step*bsize:(step+1)*bsize]
            batch_z = Z_train[step * bsize:(step + 1) * bsize]
            batch_g = G_train[step * bsize:(step + 1) * bsize]
            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             z:batch_z,g:batch_g,
                                                             keep_prob:0.7,
                                                             keep_prob_rnn:0.7})
            print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))

            step += 1
            costs.append(cost_)

        vstep = 0
        vcosts = []
        while vstep * vbsize < valid_iters:
            test_cost_ = sess.run(cost, feed_dict={x: X_valid[vstep*vbsize:(vstep+1)*vbsize],
                                                   y: Y_valid[vstep*vbsize:(vstep+1)*vbsize],
                                                   z: Z_valid[vstep*vbsize:(vstep+1)*vbsize],
                                                   g: G_valid[vstep*vbsize:(vstep+1)*vbsize],
                                                   keep_prob:1,
                                                   keep_prob_rnn: 1})
            vstep += 1
            vcosts.append(test_cost_)
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        print('valid loss: %s' %avg_cost)
        avg_cost = np.log(np.mean(np.exp(costs[-(valid_iters // bsize):])))
        print('avg train loss: %s' % avg_cost)




        X_test = load_test_data()
        Z_test = extract_charseq(X_test)
        G_test = extract_gazetter(X_test)

        num_batches = (len(X_test) // bsize) + 1


        res = np.zeros((len(X_test),6))
        for s in range(num_batches):
            print(s)
            batch_x_test = X_test[s * bsize:(s + 1) * bsize]
            batch_g_test = G_test[s * bsize:(s + 1) * bsize]
            batch_z_test = Z_test[s * bsize:(s + 1) * bsize]
            logits_ = sess.run(logits, feed_dict={x: batch_x_test,
                                                  z:batch_z_test,
                                                  g:batch_g_test,
                                                  keep_prob:1,
                                                  keep_prob_rnn:1})
            res[s * bsize:(s + 1) * bsize] = logits_

        sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
        sample_submission[list_classes] = res
        sample_submission.to_csv("submissions/XGZ_v1_e" + str(epoch) + ".csv", index=False)
