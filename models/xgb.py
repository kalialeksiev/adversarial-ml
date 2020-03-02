import numpy as np
import xgboost as xgb  # https://xgboost.readthedocs.io/en/latest/python/python_intro.html


"""
The XGBoost model is a variant of a boosted
decision tree (see: Gradient boosting) and
is very popular in machine learning competitions.

Note that XGBoost does not support categorical
features (!) so you should transform it into one-
hot encoding, instead, before training.
"""


def from_file(filename, n_threads=4, n_estimators=10,
              max_depth=4):
    bst = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_threads
    )
    bst.load_model(filename)
    return bst


def from_training_data(x, y, n_threads=4, n_estimators=10,
                       max_depth=4):
    bst = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_threads
    )
    bst.fit(x, y)
    return bst
