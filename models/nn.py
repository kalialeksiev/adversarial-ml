import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (Flatten, Dropout, Dense,
                                            BatchNormalization)
from tensorflow.python.keras.utils.np_utils import to_categorical


DEFAULT_NUM_HIDDEN_LAYERS = 3
DEFAULT_LAYER_SIZE = 1024
DEFAULT_BATCH_SIZE = 10000
DEFAULT_EPOCHS = 10


def from_file(filename):
    return tf.keras.load_model(filename)


def from_training_data(x, y, num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
                       layer_size=DEFAULT_LAYER_SIZE,
                       batch_size=DEFAULT_BATCH_SIZE,
                       epochs=DEFAULT_EPOCHS):
    y = to_categorical(y.reshape(-1, 1))

    model = Sequential()
    model.add(Dense(layer_size, input_shape=(n_features, ), activation=None))
    model.add(BatchNormalization(activation='relu'))
    model.add(Dropout(0.5))

    for i in range(num_hidden_layers):
        model.add(Dense(layer_size, activation=None))
        model.add(BatchNormalization(activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=DEFAULT_BATCH_SIZE,
              epochs=2,
              validation_split=train_test_split)
    return model
