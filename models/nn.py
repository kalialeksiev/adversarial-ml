import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import (Flatten, Dropout, Dense,
                                            BatchNormalization, Activation)
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.regularizers import l2


DEFAULT_NUM_HIDDEN_LAYERS = 4
DEFAULT_LAYER_SIZE = 1024
DEFAULT_BATCH_SIZE = 10000
DEFAULT_EPOCHS = 5
DEFAULT_OPTIMISER = 'adam'
# 600 makes the loss split equally between +ve and -ve examples, because there
# are approximately 600x as many -ve examples as +ve ones.
DEFAULT_POSITIVE_WEIGHT = 600.0


def from_file(filename):
    return tensorflow.python.keras.load_model(filename)


def from_training_data(x, y, num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
                       layer_size=DEFAULT_LAYER_SIZE,
                       batch_size=DEFAULT_BATCH_SIZE,
                       epochs=DEFAULT_EPOCHS,
                       optimiser=DEFAULT_OPTIMISER,
                       positive_weight=DEFAULT_POSITIVE_WEIGHT):
    y = to_categorical(y.reshape(-1, 1))

    # NOTE: the order of batch normalisation and dropout
    # and the activation matters; see:
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

    model = Sequential()
    # slightly bigger first layer to extract as many features as possible from the input
    model.add(Dense(2 * layer_size, input_shape=(x.shape[1],),
                    activation=None))
    model.add(BatchNormalization())  # batch norm before activation
    model.add(Activation('relu'))
    model.add(Dropout(0.2))  # less dropout for input layer

    for _ in range(num_hidden_layers):
        # add l2 regularisation
        model.add(Dense(layer_size, activation=None, kernel_regularizer=l2(1.0e-3)))
        model.add(BatchNormalization())  # batch norm before activation
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    # finally, collect together a small collection of features
    # for predicting the class probabilities in the last layer.
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser)
    model.fit(x, y, batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              class_weight={ 0 : 1.0, 1 : positive_weight})
    return model
