import pandas as pd


"""Load a pickle file containing Pandas data, and turn
it into two NumPy arrays: one of them, a 2D data matrix,
and the other a 1D target vector (which is just one of the
columns).
Optional: `cols` is a list of column names to include in the
x data matrix.
"""
def load_raw_data(filename, cols=None, target_col='isfailed'):
    
    df_co = pd.read_pickle(filename)

    x = df_co.drop(target_col, axis=1)
    if cols is not None:
        x = x[cols]  # select subset of columns
    x = x.to_numpy()

    y = df_co[target_col].to_numpy().reshape((-1,))

    return x, y


"""Convert a 2D data matrix's categorical variables to
a one-hot encoding,
"""
def convert_to_one_hot(x):
    pass
