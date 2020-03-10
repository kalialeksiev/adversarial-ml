import numpy as np
import pandas as pd
import re
from datetime import datetime
from calendar import monthrange


"""Load a pickle file containing Pandas data, and turn
it into two NumPy arrays: one of them, a 2D data matrix,
and the other a 1D target vector (which is just one of the
columns).
Optional: `cols` is a list of column names to include in the
x data matrix.
`shuffle` : should we shuffle the dataset before returning it?
"""
def load_raw_data(filename, cols=None, target_col='isfailed', shuffle=False):

    df_co = pd.read_pickle(filename)

    if shuffle:  # sample (100% of) the rows, which is effectively a shuffle
        df_co = df_co.sample(frac=1)
    
    x = df_co.drop(target_col, axis=1)
    if cols is not None:
        col_pred = get_col_matcher(cols)
        x = x[[col for col in df_co.columns if col_pred(col)]]  # select subset of columns
    x = x.to_numpy()

    y = df_co[target_col].to_numpy().reshape((-1,))

    return x, y


"""Get the accuracy of a binary (0/1) prediction.
This function returns: the true negative rate,
the true positive rate.
"""
def get_accuracy(y_true, y_pred):
    assert(y_true.shape == y_pred.shape)
    eq = (y_true == y_pred)

    true_negative = np.count_nonzero(np.logical_and(eq,
        np.logical_not(y_true)))
    true_positive = np.count_nonzero(np.logical_and(eq, y_true))

    num_negative = np.count_nonzero(np.logical_not(y_true))
    num_positive = np.count_nonzero(y_true)

    tnr = true_negative / num_negative if num_negative != 0 else 1.0
    tpr = true_positive / num_positive if num_positive != 0 else 1.0

    return (tnr, tpr)


"""Convert probabilities into a binary 0/1 value, given
a probability threshold.
"""
def threshold(y_pred, threshold):
    # apply probability threshold; this is a hacky
    # way of making y_pred equal to 1 when the probability
    # is greater than this threshold, and 0 otherwise.
    return 1 * (y_pred > threshold)


"""Creates and returns a function which, given an input string,
returns true if that string matches any of the column names, and
false otherwise. If the strings don't contain the character *, 
matching is just equality. However, * represents "any string". Thus,
the column "Field*" matches any string which starts with "Field".
"""
def get_col_matcher(cols):
    # proceed by turning the column names into a regex, then
    # return a regex checker
    expr = "|".join(cols).replace('*', '([\s\S]*)')
    re_obj = re.compile(expr)
    return lambda s: re_obj.match(s) is not None


"""Given a datetime object, return a decimal equal to the year plus
the fraction of the year the date is through (e.g. 1st Jan is 0.0,
the middle of the year is 0.5, 31st Dec is 0.99...).
"""
def date_to_decimal(dt):
    _, days_in_month = monthrange(dt.year, dt.month)
    return dt.year + (dt.month - 1) / 12.0 + (dt.day - 1) / days_in_month / 12.0


