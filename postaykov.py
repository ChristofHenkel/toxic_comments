import pandas as pd
from tensorflow.contrib.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import nltk
import tqdm
from gensim.models import KeyedVectors
import os

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")
train_data = train_data.sample(frac=1)

list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_train = train_data[list_classes].values


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict

def clear_embedding_list(model, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        row = model[word]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tqdm.tqdm(tokenized_sentences):
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train


print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

words_dict[UNKNOWN_WORD] = len(words_dict)

print("Loading embeddings...")
#embedding_list, embedding_word_dict = read_embedding_list('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec')
#embedding_size = len(embedding_list[0])

model = KeyedVectors.load_word2vec_format('assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec', binary=False)
embedding_word_dict = {w:ind for ind,w in enumerate(model.index2word)}
embedding_size = 300

print("Preparing data...")
embedding_list, embedding_word_dict = clear_embedding_list(model, embedding_word_dict, words_dict)

del model

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())
train_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    500)
test_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    500)
X_train = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)


graph = tf.Graph()

with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)

    x = tf.placeholder(tf.int32, shape=(None,500), name="input_x")
    y = tf.placeholder(tf.float32, shape=(None,6), name="input_y")
    keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")
    batch_size = tf.shape(x[0])[0]
    with tf.name_scope("Embedding"):
        #embedding = tf.get_variable("embedding", [len(tokenizer.word2index), 100], dtype=tf.float32,initializer=tf.constant_initializer(pre_embedding), trainable=False)
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=False)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    #rnn_cudnn1 = tf.contrib.cudnn_rnn.CudnnGRU(input_size= 500,num_layers= 2, num_units = 64,direction='bidirectional')
    #param_cudnn = tf.Variable(tf.zeros([rnn_cudnn1.params_size()]), validate_shape=False)
    #y_cudnn, state_cudnn = rnn_cudnn1(tf.transpose(embedded_input, [1, 0, 2]),tf.zeros([2, tf.shape(x)[0], 64]),param_cudnn)

    #rnn_cudnn2 = tf.contrib.cudnn_rnn.CudnnGRU(input_size=500, num_layers=2, num_units=64, direction='bidirectional')

    #outputs = tf.transpose(y_cudnn, [1, 0, 2])
    #outputs = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_input)
    #outputs = Dropout(keep_prob)(outputs)

    #outputs = tf.nn.dropout(outputs,keep_prob=keep_prob)
    #outputs = Bidirectional(CuDNNGRU(64, return_sequences=False))(outputs)

    with tf.variable_scope('forward'):

        fw_cell1 = tf.nn.rnn_cell.GRUCell(64,)
        fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(fw_cell1, output_keep_prob=keep_prob, seed=123)
        fw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        #fw_cell2 = tf.contrib.rnn.AttentionCellWrapper(fw_cell2,3)
        stacked_fw_rnn = [fw_cell1,fw_cell2]
        fw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_fw_rnn, state_is_tuple=True)

    with tf.variable_scope('backward'):
        bw_cell1 = tf.nn.rnn_cell.GRUCell(64)
        bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(bw_cell1, output_keep_prob=keep_prob, seed=124)
        bw_cell2 = tf.nn.rnn_cell.GRUCell(64)
        #bw_cell2 = tf.contrib.rnn.AttentionCellWrapper(bw_cell2, 3)
        stacked_bw_rnn = [bw_cell1,bw_cell2]
        bw_multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_bw_rnn, state_is_tuple=True)

    ## outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,embedded_input,dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embedded_input, dtype=tf.float32)
    output_fw, output_bw = outputs

    outputs = tf.concat([output_fw, output_bw], axis = 2)
    #outputs = tf.transpose(outputs, [0, 2, 1])
    outputs = outputs[:,:,-1]

    #outputs = tf.add(output_fw[:,:,-1],output_bw[:,:,-1])
    #outputs = tf.reduce_mean(tf.stack([output_fw[:, :, -1], output_bw[:, :, -1]]), axis=0)
    #outputs = tf.concat([output_fw[:, :, -1], output_bw[:, :, -1]], axis=0)

    #outputs = outputs[:, :, -1]
    #outputs = tf.reduce_mean(outputs, axis=2)
    x3 = layers.fully_connected(outputs, 32, activation_fn=tf.nn.relu)
    logits = layers.fully_connected(x3, 6, activation_fn=tf.nn.sigmoid)

    loss = binary_crossentropy(y,logits)
    cost = tf.losses.log_loss(predictions=logits, labels=y)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)


fold_size = len(X_train) // 10
fold_start = 0
fold_end = fold_size

train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
train_y = np.concatenate([Y_train[:fold_start], Y_train[fold_end:]])

##from mixup import augment_with_mixup

##train_x, train_y = mixup(train_x,train_y,0.7,0.3)

val_x = X_train[fold_start:fold_end]
val_y = Y_train[fold_start:fold_end]

epochs = 12
bsize = 512
train_iters = len(train_x) - bsize
steps = train_iters // bsize
valid_iters = len(val_x) - bsize

sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        costs = []
        step = 0
        while step * bsize < train_iters:
            batch_x = train_x[step*bsize:(step+1)*bsize]
            batch_y = train_y[step*bsize:(step+1)*bsize]
            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:0.7})
            print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))
            costs.append(cost_)
            step += 1

        vstep = 0
        vcosts = []
        while vstep * bsize < valid_iters:
            test_cost_ = sess.run(cost, feed_dict={x: val_x[vstep * bsize:(vstep + 1) * bsize],
                                                   y: val_y[vstep * bsize:(vstep + 1) * bsize],
                                                   keep_prob: 1
                                                   })
            vstep += 1
            vcosts.append(test_cost_)
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        print('valid loss: %s' % avg_cost)
        print('train loss %s' % np.log(np.mean(np.exp(costs[:valid_iters]))))
        num_batches = (len(X_test) // bsize) + 1

        res = np.zeros((len(X_test), 6))
        for s in range(num_batches):
            print(s)
            batch_x_test = X_test[s * bsize:(s + 1) * bsize]
            logits_ = sess.run(logits, feed_dict={x: batch_x_test,
                                                  keep_prob: 1})
            res[s * bsize:(s + 1) * bsize] = logits_


        sample_submission[list_classes] = res

        dir_name = 'pavel12/'
        if not os.path.exists('submissions/' + dir_name):
            os.mkdir('submissions/' + dir_name)
        sample_submission.to_csv("submissions/" + dir_name + "model_e"+ str(epoch) + "v"+ str(round(avg_cost,ndigits=4)) + ".csv", index=False)

