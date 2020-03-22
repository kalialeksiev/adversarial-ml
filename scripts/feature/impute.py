import pandas as pd
import numpy as np
import argparse as ap
import pickle
from sklearn.impute import SimpleImputer


"""This script will impute (fill in) all missing values
in the given dataset.
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

    print("Imputing...")

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # fit_transform returns a numpy array, but we need to keep
    # db as a pandas DataFrame. Thus we can use the little trick
    # of writing db[db.columns] = ... rather than just db = ...
    db[db.columns] = imp.fit_transform(db)

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
