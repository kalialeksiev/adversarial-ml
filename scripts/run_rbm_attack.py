import argparse as ap
import random
import numpy as np
import pandas as pd
import attacks.rbm_attack
import models.xgb
import models.rbm
import models.feature_util
import models.data_util


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

    # compute the columns to train on (only use models.feature_util.optional_cols,
    all_cols = list(db.drop(['isfailed'], axis=1).columns)
    pred = models.data_util.get_col_matcher(models.feature_util.optional_cols)
    cols = [col for col in all_cols if pred(col)]

    for k in range(args.num_companies):
        print("\nRunning on company", k + 1, "...")

        company_idx = random.choice(list(range(len(db))))  # TEMP
        company_data = db.drop(['isfailed'], axis=1).iloc[company_idx]

        result = attacks.rbm_attack.rbm_attack(model_query,
                                               rbm_model, company_data,
                                               args.target_val,
                                               cols, all_cols,
                                               threshold=args.threshold,
                                               n_restarts=args.n_restarts)

        present = company_data[cols].notna().to_numpy().reshape((1, -1))
        x = company_data.to_numpy().reshape((1, -1))
        y = np.copy(x)

        corrupt_cols = []
        for i in range(result.shape[-1]):
            if result[0][i] != present[0][i]:
                corrupt_cols.append(cols[i])
                # which column does this correspond to in the original table?
                j = list(db.columns).index(cols[i])
                y[0][j] = np.NaN
        
        print("Corrupt the following columns:", corrupt_cols)
        print("-->", "changes model value from", model_query(x), "to",
            model_query(y))
        print("-->", "changes input likelihood from", np.exp(rbm_model.score_samples(present)), "to",
            np.exp(rbm_model.score_samples(result)))
