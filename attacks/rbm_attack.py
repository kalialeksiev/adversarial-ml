import random
import pandas as pd
import numpy as np
import models.feature_util
import models.rbm as rbm


"""Perform the restricted Boltzmann machine attack
on the given model. This attack is basically for trying
to achieve a particular prediction by removing existing
data about the company.

model_query: a function which takes as input a row of data,
             and returns the probability of failure for that
             row.
rbm_model: a trained RBM model (from models.rbm)
company_data: a row of data for a given company (to be passed
              directly to both models) as a Pandas row.
target_val: either 1.0 or 0.0 (try to make the model predict
            isfailed==target_val)
optional_cols: column names that we are allowed to modify
n_restarts: the number of times to repeat the whole procedure (to
            bootstrap the probability of fooling the model).
threshold: we will not modify the data if it brings the likelihood
           of being true below this probability.

returns: a boolean mask over the columns in the company data
         which represents whether or not they should be present
"""
def rbm_attack(model_query, rbm_model, company_data, target_val,
               optional_cols, all_cols, n_restarts=10, threshold=0.5):
    # check other inputs
    assert(target_val == 1.0 or target_val == 0.0)
    assert(len(optional_cols) <= len(all_cols))
    assert(len(all_cols) == company_data.shape[-1])

    # determine the indices of the columns we are allowed to modify
    col_indices = [all_cols.index(col) for col in optional_cols]
    
    best = None

    for _ in range(n_restarts):
        # at every restart iteration, restore 'x' and 'present':
        # convert company row to numpy array
        x = company_data.to_numpy().reshape((1, -1))

        # a boolean vector representing whether data is present or not
        present = company_data[optional_cols].notna().to_numpy().reshape((1, -1))

        if best is None:
            best = present  # initialise 'best' with default value

        p_failed = model_query(x)
        p_true = np.exp(rbm_model.score_samples(present))

        # shuffle indices to try corrupting them in a random order
        random.shuffle(col_indices)

        # try corrupting the columns
        for j, i in enumerate(col_indices):
            if present[0][j]:
                # try corrupting ith column
                x_test = np.copy(x)
                present_test = np.copy(present)
                x_test[0][i] = None
                present_test[0][j] = False
                # compute probabilities for this test value:
                p_failed_test = model_query(x_test)
                p_true_test = np.exp(rbm_model.score_samples(present_test))
                # is this feasible and better than our current estimate?
                if (abs(p_failed_test - target_val) < abs(p_failed - target_val)
                    and p_true_test >= threshold):
                    x = x_test
                    present = present_test
                    p_failed = p_failed_test
                    p_true = p_true_test
                    best = present  # update this
    
    return best
