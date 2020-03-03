import numpy as np
import pandas as pd


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
        x = x[cols]  # select subset of columns
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
