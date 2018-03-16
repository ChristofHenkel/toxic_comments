import re, os, gc, time, pandas as pd, numpy as np
import tqdm

np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "5"
from nltk import tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, Add, Flatten, TimeDistributed,CuDNNGRU,CuDNNLSTM
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
# from keras.engine.topology import Layer
from keras.engine import InputSpec, Layer
from global_variables import TRAIN_FILENAME, TEST_FILENAME, COMMENT, LIST_CLASSES
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


"""
I should also try:
https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py
"""


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


embed_size = 300
max_features = 150000
max_text_len = 150
max_sent = 5

# EMBEDDING_FILE = "../input/glove840b300dtxt/glove.840B.300d.txt
EMBEDDING_FILE = "assets/embedding_models/ft_300d_crawl/crawl-300d-2M.vec"


def clean_corpus(comment):
    comment = comment.replace('&', ' and ')
    comment = comment.replace('0', ' zero ')
    comment = comment.replace('1', ' one ')
    comment = comment.replace('2', ' two ')
    comment = comment.replace('3', ' three ')
    comment = comment.replace('4', ' four ')
    comment = comment.replace('5', ' five ')
    comment = comment.replace('6', ' six ')
    comment = comment.replace('7', ' seven ')
    comment = comment.replace('8', ' eight ')
    comment = comment.replace('9', ' nine ')
    comment = comment.replace('\'ve', ' have ')
    comment = comment.replace('\'d', ' would ')
    comment = comment.replace('\'m', ' am ')
    comment = comment.replace('n\'t', ' not ')
    comment = comment.replace('\'s', ' is ')
    comment = comment.replace('\'r', ' are ')
    comment = re.sub(r"\\", "", comment)
    comment = word_tokenize(comment)
    comment = " ".join(word for word in comment)
    return comment.strip().lower()


tic = time.time()

train = pd.read_csv(TRAIN_FILENAME)
test = pd.read_csv(TEST_FILENAME)
Y = train[LIST_CLASSES].values

print('cleaning corpus')
train[COMMENT].fillna("no comment", inplace = True)
train[COMMENT] = train[COMMENT].apply(lambda x: clean_corpus(x))

test[COMMENT].fillna("no comment", inplace = True)
test[COMMENT] = test[COMMENT].apply(lambda x: clean_corpus(x))

print('tokenizing')
train["sentences"] = train[COMMENT].apply(lambda x: tokenize.sent_tokenize(x))
test["sentences"] = test[COMMENT].apply(lambda x: tokenize.sent_tokenize(x))
toc = time.time()
print(toc-tic)


from keras.preprocessing.text import Tokenizer, text_to_word_sequence

print('fitting tokenizer')
raw_text = train[COMMENT]
tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text)

def sentenize(data):
    comments = data["sentences"]
    sent_matrix = np.zeros((len(comments), max_sent, max_text_len), dtype = "int32")
    for i, sentences in enumerate(comments):
        for j, sent in enumerate(sentences):
            if j < max_sent:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_text_len and tk.word_index[word] < max_features:
                            sent_matrix[i, j, k] = tk.word_index[word]
                            k = k+1
                    except:
                            sent_matrix[i, j, k] = 0
                            k = k+1
    return sent_matrix

print('sentenizing')
X = sentenize(train)
X_test = sentenize(test)

del train, test
gc.collect()

print('loading embeddings')
tic = time.time()
def get_coefs(word,*arr): return word, np.asarray(arr, dtype = "float32")
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
toc = time.time()
print(toc-tic)


from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model(rnn_units = 100, lr = 0.0):
    sentence_input = Input(shape = (max_text_len,), dtype = "int32")
    embedded_sequences = Embedding(nb_words, embed_size, weights=[embedding_matrix],
              input_length=max_text_len, trainable=False)(sentence_input)

    l_lstm = Bidirectional(LSTM(rnn_units))(embedded_sequences)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape = (max_sent, max_text_len), dtype = "int32")
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(rnn_units))(review_encoder)

    out = Dense(6, activation = "sigmoid")(l_lstm_sent)
    model = Model(review_input, out)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr),  metrics = ["accuracy"])
    return model



model = build_model(rnn_units = 64, lr = 1e-3)
model.summary()

print("model fitting - Hierachical LSTM")
#fold_count = 10
fold_count = 1
fold_size = len(X) // 10
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    model = build_model(rnn_units = 64, de_units = 64, lr = 1e-3)
    file_path = "HAN_%s_.hdf5" %fold_id
    ra_val = RocAucEvaluation(validation_data = (X_valid, Y_valid), interval = 1)
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 3, validation_data = (X_valid, Y_valid),
                    verbose = 1, callbacks = [ra_val, check_point])


list_of_preds = []
list_of_vals = []
list_of_y = []
fold_count = 10
fold_size = len(X) // 10
for fold_id in range(0, fold_count):
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_valid = X[fold_start:fold_end]
    Y_valid = Y[fold_start:fold_end]
    X_train = np.concatenate([X[:fold_start], X[fold_end:]])
    Y_train = np.concatenate([Y[:fold_start], Y[fold_end:]])

    file_path = 'HAN_' + str(fold_id) + '_.hdf5'
    model = load_model(file_path, custom_objects = {"AttentionWeightedAverage": AttentionWeightedAverage})
    preds = model.predict(X_test, batch_size = 1024, verbose = 1)
    list_of_preds.append(preds)
    vals = model.predict(X_valid, batch_size = 1024, verbose = 1)
    list_of_vals.append(vals)
    list_of_y.append(Y_valid)

test_predicts = np.zeros(list_of_preds[0].shape)
for fold_predict in list_of_preds:
    test_predicts += fold_predict

test_predicts /= len(list_of_preds)
submission = pd.read_csv('sample_submission.csv')
submission[LIST_CLASSES] = test_predicts
submission.to_csv('l2_test_data.csv', index=False)

l2_data = pd.DataFrame(columns=['logits_' + c for c in LIST_CLASSES] + LIST_CLASSES)
l2_data[['logits_' + c for c in LIST_CLASSES]] = pd.DataFrame(np.concatenate(list_of_vals, axis = 0))
l2_data[LIST_CLASSES] = pd.DataFrame(np.concatenate(list_of_y, axis = 0))
l2_data.to_csv('l2_train_data.csv')