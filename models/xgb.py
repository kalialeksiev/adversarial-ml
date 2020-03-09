import numpy as np
import xgboost as xgb  # https://xgboost.readthedocs.io/en/latest/python/python_intro.html


"""
The XGBoost model is a variant of a boosted
decision tree (see: Gradient boosting) and
is very popular in machine learning competitions.

Note that XGBoost does not support categorical
features (!) so you should transform it into one-
hot encoding, instead, before training.

Parameters for the XGBoost model:

n_threads : the number of threads to use for training/prediction
n_estimators : the number of gradient-boosted trees (equivalent
               to the number of boosting rounds)
max_depth : the maximum tree depth for base learners
learning_rate : equivalent to XGB's "eta"
"""


DEFAULT_NUM_THREADS = 4
DEFAULT_NUM_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 3
DEFAULT_LEARNING_RATE = 1.0e-2


def from_file(filename, n_threads=DEFAULT_NUM_THREADS,
              n_estimators=DEFAULT_NUM_ESTIMATORS,
              max_depth=DEFAULT_MAX_DEPTH,
              learning_rate=DEFAULT_LEARNING_RATE):
    bst = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_threads,
        learning_rate=learning_rate
    )
    bst.load_model(filename)
    return bst


def from_training_data(x, y, n_threads=DEFAULT_NUM_THREADS,
                       n_estimators=DEFAULT_NUM_ESTIMATORS,
                       max_depth=DEFAULT_MAX_DEPTH,
                       learning_rate=DEFAULT_LEARNING_RATE):
    bst = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_threads,
        learning_rate=learning_rate
    )
    bst.fit(x, y)
    return bst
