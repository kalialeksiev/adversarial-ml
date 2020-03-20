import random
import pandas as pd
import numpy as np
import models.feature_util
import models.data_util
import models.rbm as rbm
from attacks.rbm_attack_result import RBMAttackResult


def eval_corruption(model_query, rbm_model, company_data, rbm_attack_result):
    x = company_data.copy()
    rbm_attack_result.apply_corruption(x)
    present = company_data.copy()
    rbm_attack_result.detect_corruption(present)

    # convert both to numpy
    x = x.to_numpy()
    present = present.to_numpy()

    # run models
    p_failed = model_query(x)[0]
    p_true = np.exp(rbm_model.score_samples(present))

    return p_failed, p_true


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
rbm_attack_result : an RBMAttackResult object which is initialised
                    with all the columns necessary.
n_restarts: the number of times to repeat the whole procedure (to
            bootstrap the probability of fooling the model).
threshold: we will not modify the data if it brings the likelihood
           of being true below this probability.

returns: a new RBMAttackResult object which contains information
         about the corrupted columns.
"""
def rbm_attack(model_query, rbm_model, company_data, target_val,
               rbm_attack_result, n_restarts=10, threshold=0.5):
    # check other inputs
    assert(target_val == 1.0 or target_val == 0.0)

    # determine the names and indices of the columns we are allowed to modify
    optional_cols, col_indices = rbm_attack_result.get_corruptible(return_indices=True)

    best = rbm_attack_result
    p_failed, p_true = eval_corruption(model_query, rbm_model,
                                        company_data, rbm_attack_result)

    for _ in range(n_restarts):
        # for each restart iteration, start fresh
        attack_result = rbm_attack_result

        # shuffle columns so as to corrupt them in a random order
        random.shuffle(optional_cols)

        for col in optional_cols:
            # corrupt this column
            attack_result = attack_result.corrupt_col(col)
            # evaluate current set of corrupted columns
            (p_failed_test,
             p_true_test) = eval_corruption(model_query, rbm_model,
                                            company_data, attack_result)

            if p_true_test < threshold:
                break  # this example has failed, try again on the next restart

            # is this feasible and better than our current best estimate?
            if abs(p_failed_test - target_val) < abs(p_failed - target_val):
                p_failed = p_failed_test
                p_true = p_true_test
                best = attack_result  # store our new best attack result

    return best


"""Gets the columns for which we will attempt the RBM attack on.
Returns two lists: the list of column names to attack, and a list
containing corresponding indicator columns if applicable (e.g. FieldN
and hasFN.)
"""
def get_rbm_attack_columns(all_cols):
    # for now, only try corrupting the accounting fields and date columns
    pred = models.data_util.get_col_matcher(models.feature_util.date_cols)
    date_cols = [col for col in all_cols if pred(col)]
    optional_cols = ['Field' + str(n)
                     for n in models.feature_util.accounting_field_nums] + date_cols
    indicator_cols = [('hasF' + str(n) if 'hasF' + str(n) in all_cols else None)
                      for n in models.feature_util.accounting_field_nums
                      ] + [None] * len(date_cols)
    return optional_cols, indicator_cols
