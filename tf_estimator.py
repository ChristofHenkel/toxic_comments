"""
https://www.tensorflow.org/programmers_guide/estimators
https://www.tensorflow.org/get_started/custom_estimators
"""

import tensorflow as tf
import tqdm
import numpy as np

# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None,classes=6)
#keras_xception = tf.keras.applications.xception.Xception(weights=None,classes=6)
#k = tf.keras.applications.vgg16.VGG16(classes=6)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
#keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
#                          loss='binary_crossentropy',
#                          metric='accuracy')
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),loss='binary_crossentropy',metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:

train_data = np.array([[embedding_matrix[v][:100] for v in x] for x in tqdm.tqdm(X_train[:10000])],dtype=np.float32)
train_data = np.expand_dims(train_data, axis=3)
train_data = np.repeat(train_data, 3, axis=3)
train_labels = Y_train[:10000]
train_labels = train_labels.astype(np.float32)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_2": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False,
batch_size=32)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
