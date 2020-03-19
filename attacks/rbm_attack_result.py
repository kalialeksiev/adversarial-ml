import pandas as pd
import numpy as np
import models.feature_util
import copy


"""This class encapsulates the result of an RBM attack, and basically
stores which columns are corrupted as a result of the attack.
However there is more to it than that - because some columns have
corresponding indicator columns, e.g. "FieldN" and the corresponding
"hasFN". Thus when we corrupt one column we need to update its indicator.

! This class is immutable ! (Any operations which modify its state instead
return a copy of the object with the modifications applied.)
"""
class RBMAttackResult:
    """
    all_cols : all of the columns in the database
    optional_cols : the subset of columns we are allowed to corrupt
    indicator_cols : the indicator columns corresponding to each optional
                     column (use None if the optional column does not have
                     an indicator column.)
    """
    def __init__(self, all_cols, optional_cols, indicator_cols):
        assert(len(optional_cols) <= len(all_cols))
        assert(len(optional_cols) == len(indicator_cols))

        self.all_cols = list(all_cols)
        self.optional_cols = list(optional_cols)
        self.indicator_cols = list(indicator_cols)
        self.corrupted = []

    # get the list of columns the attack is allowed to corrupt
    # return_indices: whether or not to return the indices of the
    #                 columns as well as their names.
    def get_corruptible(self, return_indices=False):
        if return_indices:
            return (self.optional_cols,
                    [self.all_cols.index(col)
                     for col in self.optional_cols])
        else:
            return self.optional_cols

    # get columns corrupted so far
    def get_corrupted(self):
        return self.corrupted

    # add a column to the corrupted list
    def corrupt_col(self, col):
        result = copy.deepcopy(self)
        result.corrupted.append(col)
        return result

    # based on the current corruption state in this object, corrupt
    # the given data; "rows" should be a pandas DataFrame.
    # Modifies "rows" directly!
    def apply_corruption(self, rows):
        assert(rows.shape[0] == len(self.all_cols))

        # get the indicator column names of the corrupted columns
        corrupted_indicators = [self.indicator_cols[
            self.optional_cols.index(col)
        ] for col in self.corrupted]
        corrupted_indicators = [col for col in corrupted_indicators if col is not None]

        # apply the corruption!
        rows[self.corrupted] = np.NaN
        rows[corrupted_indicators] = 0

    # based on the current corruption state in this object, turn
    # a set of data into a table indicating, for each optional (corruptible)
    # column, whether or not it is present. A piece of data is not present
    # if and only if it was not present initially, or if it was in a corrupted
    # column. Modifies "rows" directly! "rows" should be a pandas DataFrame.
    def detect_corruption(self, rows):
        assert(rows.shape[0] == len(self.all_cols))

        rows[self.corrupted] = np.NaN
        not_corrupted_cols = [col for col in self.all_cols if not col in self.corrupted]
        rows.drop(not_corrupted_cols, axis=1, inplace=True)
        rows[self.corrupted] = rows[self.corrupted].notna()
            

