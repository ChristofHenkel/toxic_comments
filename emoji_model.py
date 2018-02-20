import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from global_variables import TRAIN_FILENAME

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
batch_size = 512






train = pd.read_csv(TRAIN_FILENAME)
features = pd.read_csv('assets/raw_data/train_emoji_features.csv')

Y = train[label_cols].values

#X = train['features'].values
#X = [np.fromstring(a[1:-2],sep = ',') for a in tqdm(X)]
#X = np.array(X)
X = features.values
X = X[:,1:]

split_at = len(X) // 10

X_train = X[split_at:]
X_valid = X[:split_at]
Y_train = Y[split_at:]
Y_valid = Y[:split_at]
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(shape=(None, 2304), name='x', dtype=tf.float32)
    y = tf.placeholder(shape=(None, 6), name='y', dtype=tf.float32)
    keep_prob = tf.placeholder(dtype=tf.float32)


    #with tf.variable_scope('hidden'):
    #    h1 = layers.fully_connected(x, 16, activation_fn=tf.nn.elu)
    #    h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

    logits = layers.fully_connected(x, 6, activation_fn=tf.nn.sigmoid)
    # loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,logits=logits)
    loss = binary_crossentropy(y, logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

    (_, roc_auc_op) = tf.metrics.auc(labels=y, predictions=logits)

num_batches = len(X_train) // batch_size + 1
epochs = 15

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(epochs):
        tf.local_variables_initializer().run(session=sess)
        for s in range(num_batches):
            batch_x = X_train[s * batch_size:(s + 1) * batch_size]
            batch_y = Y_train[s * batch_size:(s + 1) * batch_size]

            roc, _ = sess.run([roc_auc_op, optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
            # print(l)

        print(roc)

        val_iters = len(X_valid) // batch_size + 1
        tf.local_variables_initializer().run(session=sess)
        for v in range(val_iters):
            batch_x_val = X_valid[v * batch_size:(v + 1) * batch_size]
            batch_y_val = Y_valid[v * batch_size:(v + 1) * batch_size]

            roc_val = sess.run(roc_auc_op, feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1})

        # print(l_vals / val_iters)
        print(roc_val)



## TRY SVM



