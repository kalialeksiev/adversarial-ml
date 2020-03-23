import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (Flatten, Dropout, Dense,
                                            BatchNormalization, Activation)
from tensorflow.python.keras.utils.np_utils import to_categorical


DEFAULT_NUM_HIDDEN_LAYERS = 4
DEFAULT_LAYER_SIZE = 1024
DEFAULT_BATCH_SIZE = 10000
DEFAULT_EPOCHS = 5
DEFAULT_OPTIMISER = 'adam'
# 600 makes the loss split equally between +ve and -ve examples, because there
# are approximately 600x as many -ve examples as +ve ones.
DEFAULT_POSITIVE_WEIGHT = 600.0


def from_file(filename):
    return tf.keras.load_model(filename)


def from_training_data(x, y, num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
                       layer_size=DEFAULT_LAYER_SIZE,
                       batch_size=DEFAULT_BATCH_SIZE,
                       epochs=DEFAULT_EPOCHS,
                       optimiser=DEFAULT_OPTIMISER,
                       positive_weight=DEFAULT_POSITIVE_WEIGHT):
    y = to_categorical(y.reshape(-1, 1))

    # note that the order of batch normalisation and dropout
    # matters; see:
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

    model = Sequential()
    model.add(Dense(2 * layer_size, input_shape=(x.shape[1],),
                    activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for i in range(num_hidden_layers):
        model.add(Dense(layer_size, activation=None))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser)
    model.fit(x, y, batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              class_weight={ 0 : 1.0, 1 : positive_weight})
    return model
