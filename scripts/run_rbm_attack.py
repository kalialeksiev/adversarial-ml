import argparse
import attacks.rbm_attack


"""This script will perform the RBM attack to particular
rows of the dataset to determine if we can fool the model
only by removing information about the company.
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

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

    args = parser.parse_args()


