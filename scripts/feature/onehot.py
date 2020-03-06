import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import KernelPCA
import argparse as ap
import pickle
from models.feature_util import categorical_cols


"""
This script will convert categorical columns into one-hot features.
Warning: this adds a lot of extra columns.
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")

    args = parser.parse_args()

    print("Loading input...")

    db = pd.read_pickle(args.source)

    print("One-hotting categorical columns...")

    # thankfully, Pandas has a function that does this for us:
    db = pd.get_dummies(db, prefix=categorical_cols,
                        columns=categorical_cols)

    print("Done. Saving...")

    pickle.dump(db, open(args.dest, "wb"))
