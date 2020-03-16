import pickle
import argparse as ap
import pandas as pd
import attacks.rbm_attack
import models.rbm
import models.feature_util
import models.data_util


"""This script will train the RBM model for usage in the RBM
attack.
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('--out_model', type=str,
        default="data/rbm_model.bin",
        help="Path to the location to store the trained model.")
    parser.add_argument('data', type=str,
        help="Path to data file to train the RBM model, in Pickle format.")
    parser.add_argument('--n_components', type=int,
                        default=models.rbm.DEFAULT_N_COMPONENTS,
                        help="Number of latent variables in RBM model.")
    parser.add_argument('--n_iters', type=int,
                        default=models.rbm.DEFAULT_N_ITERS,
                        help="Number of passes over training data during training.")
    parser.add_argument('--learning_rate', type=float,
                        default=models.rbm.DEFAULT_LEARNING_RATE,
                        help="Learning rate of training.")

    args = parser.parse_args()

    print("Loading data...")

    db = pd.read_pickle(args.data)

    # compute the columns to train on (only use models.feature_util.optional_cols,
    pred = models.data_util.get_col_matcher(models.feature_util.optional_cols)
    cols = [col for col in db.drop(['isfailed'], axis=1).columns if pred(col)]

    # compute training data from db (.notna() will turn them into booleans
    # indicating whether or not the fields are present, which is what we're
    # wanting to train on.)
    x = db[cols].notna()

    print("Training model...")

    rbm_model = models.rbm.from_training_data(x, n_components=args.n_components,
                                              learning_rate=args.learning_rate,
                                              n_iters=args.n_iters)
    
    print("Done. Saving model...")

    pickle.dump(rbm_model, open(args.out_model, "wb"))
