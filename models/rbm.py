import numpy as np
import pickle
from sklearn.neural_network import BernoulliRBM


"""Restricted Boltzmann machines are an unsupervised model
for Bernoulli inputs which estimate the distribution of the
inputs. They are used in the RBM attack.
"""


DEFAULT_N_COMPONENTS = 100  # hidden dimension
DEFAULT_LEARNING_RATE = 1.0e-2
DEFAULT_N_ITERS = 10  # num iterations to do over the dataset during training


def from_file(filename):
    rbm = pickle.load(open(filename, "rb"))

    return rbm


def from_training_data(x, n_components=DEFAULT_N_COMPONENTS,
                       n_iters=DEFAULT_N_ITERS,
                       learning_rate=DEFAULT_LEARNING_RATE):

    rbm = BernoulliRBM(n_components=n_components,
                    learning_rate=learning_rate,
                    n_iter=n_iters,
                    random_state=0)  # random state = 0 just fixes the seed

    rbm.fit(x)

    return rbm


