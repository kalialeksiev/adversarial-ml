import argparse as ap
import random
import numpy as np
import pandas as pd
import attacks.rbm_attack
import models.xgb
import models.rbm
import models.feature_util
import models.data_util
from attacks.rbm_attack_result import RBMAttackResult


"""This script will perform the RBM attack to particular
rows of the dataset to determine if we can fool the model
only by removing information about the company.
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('num_companies', type=int,  # TEMP
                        help="The number of randomly chosen companies to attack.")
    parser.add_argument('target_val', type=float,
                        help="Choose either 0 or 1 - this is the value we will try "
                             "to trick the model into predicting.")
    parser.add_argument('--xgb_model', type=str,
        default="data/xgb_model.bin",
        help="Path to the trained XGBoost model.")
    # TODO: add neural network model argument here
    parser.add_argument('--rbm_model', type=str,
        default="data/rbm_model.bin",
        help="Path to the trained RBM model.")
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help="Minimum probability threshold for the attack (lower means "
                             "we're allowed to remove more data, higher means we need to "
                             "keep the data looking more realistic.)")
    parser.add_argument('--n_restarts', type=int,
                        default=10,
                        help="Number of times to attempt the attack procedure ("
                             "increasing this just increases the chance of fooling "
                             "the model.)")
    # TODO: how to input the company data for us to tweak? How to automatically clean?
    # (for now I have just loaded the full database below and selected a random row.)

    args = parser.parse_args()

    print("Loading models and data...")

    xgb_model = models.xgb.from_file(args.xgb_model)
    def model_query(x):
        return xgb_model.predict_proba(x).T[1]

    rbm_model = models.rbm.from_file(args.rbm_model)

    db = pd.read_pickle("data/clean.pkl")  # TEMP

    # compute the columns to train on, then put this info into an RBMAttackResult object.
    all_cols = list(db.drop(['isfailed'], axis=1).columns)
    pred = models.data_util.get_col_matcher(models.feature_util.date_cols)
    date_cols = [col for col in all_cols if pred(col)]
    optional_cols = ['Field' + str(n)
                     for n in models.feature_util.accounting_field_nums] + date_cols
    indicator_cols = [('hasF' + str(n) if 'hasF' + str(n) in list(db.columns) else None)
                      for n in models.feature_util.accounting_field_nums
                      ] + [None] * len(date_cols)
    initial_attack_result = RBMAttackResult(all_cols, optional_cols, indicator_cols)

    for k in range(args.num_companies):
        print("\nRunning on company", k + 1, "...")

        company_idx = random.choice(list(range(len(db))))  # TEMP
        company_data = db.drop(['isfailed'], axis=1).iloc[company_idx]

        result = attacks.rbm_attack.rbm_attack(model_query,
                                               rbm_model, company_data,
                                               args.target_val,
                                               initial_attack_result,
                                               threshold=args.threshold,
                                               n_restarts=args.n_restarts)
        
        if len(result.get_corrupted()) > 0:
            p_prior_failed, p_prior_true = attacks.rbm_attack.eval_corruption(model_query,
                rbm_model, company_data, initial_attack_result
            )
            p_posterior_failed, p_posterior_true = attacks.rbm_attack.eval_corruption(model_query,
                rbm_model, company_data, result
            )

            print("-->", "corrupt the following columns:", result.get_corrupted())

            print("-->", "changes model value from", p_prior_failed, "to",
                  p_posterior_failed, "( change of", p_posterior_failed - p_prior_failed, ")")
            print("-->", "changes input likelihood from", p_prior_true, "to",
                  p_posterior_true)
            assert(p_prior_failed != p_posterior_failed)
        else:
            print("-->", "could not change model's output.")
