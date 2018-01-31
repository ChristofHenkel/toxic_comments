import pandas as pd
import numpy as np
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
import tensorflow as tf
from tensorflow.contrib import layers
from utilities import get_oov_vector
import nltk
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
from preprocess_utils import Preprocessor
import tqdm
import os
import time

from mixup import augmented_with_translation, mixup

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
MAXLEN = 500

train_data = pd.read_csv("assets/raw_data/train.csv")
test_data = pd.read_csv("assets/raw_data/test.csv")
preprocessor = Preprocessor()

#train_data = augmented_with_translation_disk(train_data, 0.3)


sentences_train = train_data["comment_text"].fillna("_NAN_").values
#sentences_valid = valid_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

Y = train_data[list_classes].values
#Y_valid = valid_data[list_classes].values

print(sentences_train[3])

def tokenize_sentences(sentences, words_dict, mode = 'nltk'):
    twitter_tokenizer = TweetTokenizer()
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        sentence = preprocessor.expand_contractions(sentence)
        if mode == 'nltk':
            tokens = nltk.tokenize.word_tokenize(sentence)
        elif mode == 'twitter':
            tokens = twitter_tokenizer.tokenize(sentence)
        else:
            tokens = None
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
    k = 0
    l = 0
    for word in tqdm.tqdm(words_dict):
        if word not in embedding_word_dict:
            l += 1
            row = get_oov_vector(word,model,threshold=0.7)
            if row is None:
                k += 1
                continue
            else:
                cleared_embedding_list.append(row)
                cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
        else:
            row = model[word]
            cleared_embedding_list.append(row)
            cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)
    print('embeddings not found: {0:.1f}%'.format(l / len(words_dict) * 100))
    print('embeddings not synthesized: {0:.1f}%'.format(k/len(words_dict)*100))
    return cleared_embedding_list, cleared_embedding_word_dict

def get_bad_sentences(vlosses, vlogits, X_valid, Y_valid):
    idx = (-vlosses).argsort()[:100]
    X = X_valid[idx]
    Y = Y_valid[idx]
    preds = np.concatenate((Y,vlogits[idx]))
    losses = vlosses[idx]
    sentences = []
    for row in X:
        sentences.append(' '.join([id_to_embedded_word[r] for r in row]))
    d = pd.DataFrame(preds, columns=list_classes.extend(['l' + label for label in list_classes]))
    #d[list_classes] = Y
    d['words'] = pd.Series(sentences)
    d['idx'] = pd.Series(idx)
    d['loss'] = pd.Series(losses)
    d.to_csv('misclassifies2.csv', index=False)

def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tqdm.tqdm(tokenized_sentences,mininterval=5):
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
tokenized_sentences_train, words_dict = tokenize_sentences(sentences_train, {}, mode='twitter')

#print("Tokenizing sentences in validation set...")
#tokenized_sentences_valid, words_dict = tokenize_sentences(sentences_valid, words_dict)

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(sentences_test, words_dict, mode='twitter')

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
id_to_embedded_word = dict((id, word) for word, id in embedding_word_dict.items())
train_list_of_token_ids = convert_tokens_to_ids(tokenized_sentences_train,id_to_word,embedding_word_dict,MAXLEN)
#valid_list_of_token_ids = convert_tokens_to_ids(tokenized_sentences_valid,id_to_word,embedding_word_dict,MAXLEN)

test_list_of_token_ids = convert_tokens_to_ids(tokenized_sentences_test,id_to_word,embedding_word_dict,MAXLEN)

X = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)


fold_size = len(X) // 10
models = []

fold_start = 0
fold_end = fold_start + fold_size

X_valid = X[fold_start:fold_end]
Y_valid = Y[fold_start:fold_end]
X_train = np.concatenate([X[:fold_start], X[fold_end:]])
Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])



#X_train, Y_train = mixup(X_train,Y_train,2,0.1, seed=23)



bsize = 512


graph = tf.Graph()

with graph.as_default():
    # tf Graph input
    tf.set_random_seed(1)

    x = tf.placeholder(tf.int32, shape=(None,MAXLEN), name="input_x")
    y = tf.placeholder(tf.float32, shape=(None,6), name="input_y")
    keep_prob = tf.placeholder(dtype=tf.float32, name="input_keep_prob")
    with tf.name_scope("Embedding"):
        #embedding = tf.get_variable("embedding", [len(tokenizer.word2index), 100], dtype=tf.float32,initializer=tf.constant_initializer(pre_embedding), trainable=False)
        embedding = tf.get_variable("embedding", [embedding_matrix.shape[0], embedding_matrix.shape[1]], dtype=tf.float32,initializer=tf.constant_initializer(embedding_matrix), trainable=True)
        embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    """
    fw_cudnn_cell1 = tf.contrib.cudnn_rnn.CudnnGRU(input_size= 500,num_layers= 1, num_units = 64,direction='bidirectional',seed = 123)
    param_cudnn = tf.Variable(tf.zeros([fw_cudnn_cell1.params_size()]), validate_shape=False)
    y_cudnn, state_cudnn = fw_cudnn_cell1(tf.transpose(embedded_input, [1, 0, 2]),tf.zeros([2, bsize, 64]),param_cudnn)
    outputs = tf.transpose(y_cudnn, [1, 0, 2])
    outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
    fw_cudnn_cell2 = tf.contrib.cudnn_rnn.CudnnGRU(input_size= 500,num_layers= 1, num_units = 64,direction='bidirectional',seed = 123)
    param_cudnn2 = tf.Variable(tf.zeros([fw_cudnn_cell2.params_size()]), validate_shape=False)
    y_cudnn2, state_cudnn2 = fw_cudnn_cell2(tf.transpose(outputs, [1, 0, 2]),tf.zeros([2, bsize, 64]),param_cudnn2)
    outputs = tf.transpose(y_cudnn2, [1, 0, 2])
    """



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

    loss = binary_crossentropy(y,logits)
    cost = tf.losses.log_loss(predictions=logits, labels=y)
    #loss = tf.losses.sigmoid_cross_entropy(y,logits)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)





epochs = 12
train_iters = len(X_train) - bsize
steps = train_iters // bsize
valid_iters = len(X_valid) - bsize

sample_submission = pd.read_csv("assets/raw_data/sample_submission.csv")

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(epochs):
        tic = time.time()
        costs = []
        step = 0
        while step * bsize < train_iters:
            batch_x = X_train[step * bsize:(step + 1) * bsize]
            batch_y = Y_train[step * bsize:(step + 1) * bsize]
            cost_ , _ = sess.run([cost,optimizer],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             keep_prob:0.7})
            if step % 10 == 0:
                print('e %s/%s  --  s %s/%s  -- cost %s' %(epoch,epochs,step,steps,cost_))
            costs.append(cost_)
            step += 1

        vstep = 0
        vcosts = []
        vlosses = np.asarray([])
        while vstep * bsize < valid_iters:
            test_cost_, valid_loss = sess.run([cost,loss], feed_dict={x: X_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   y: Y_valid[vstep * bsize:(vstep + 1) * bsize],
                                                   keep_prob: 1
                                                   })
            vstep += 1
            vcosts.append(test_cost_)
            vlosses = np.concatenate((vlosses,valid_loss))
        avg_cost = np.log(np.mean(np.exp(vcosts)))
        toc = time.time()
        print('time needed %s' %(toc-tic))
        print('valid loss: %s' % avg_cost)
        avg_train_cost = np.log(np.mean(np.exp(costs[:valid_iters])))
        print('train loss %s' %avg_train_cost )



        num_batches = (len(X_test) // bsize) + 1
        res = np.zeros((len(X_test), 6))
        for s in range(num_batches):
            if s % 50 == 0:
                print(s)
            batch_x_test = X_test[s * bsize:(s + 1) * bsize]
            #if s == num_batches - 1:
            #    pad_size = bsize - batch_x_test.shape[0]
            #    pad = np.zeros(shape=(pad_size, X_test.shape[1]))
            #    batch_x_test = np.concatenate((batch_x_test, pad))
            logits_ = sess.run(logits, feed_dict={x: batch_x_test,
                                                  keep_prob: 1})
            #if s != num_batches - 1:
            res[s * bsize:(s + 1) * bsize] = logits_
            #else:
            #    res[s * bsize:(s + 1) * bsize - pad_size] = logits_[:bsize - pad_size]

        sample_submission[list_classes] = res

        dir_name = 'pavel30/'
        if not os.path.exists('submissions/' + dir_name):
            os.mkdir('submissions/' + dir_name)
        fn = "submissions/"
        fn += dir_name + "model_e"+ str(epoch)
        fn += "v"+ str(round(avg_cost,ndigits=4)) + "t"+ str(round(avg_train_cost,ndigits=4)) + ".csv"
        sample_submission.to_csv(fn, index=False)


