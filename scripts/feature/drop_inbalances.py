import pandas as pd
import numpy as np
import argparse as ap
import pickle



"""This script will drop features with only two values
present, and where one of the values is taken by over
99.9% of the data.
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")
    parser.add_argument('--threshold', type=float,
        default=0.999,
        help="The minimum proportion required to drop.")

    args = parser.parse_args()

    print("Loading input...")

    db = pd.read_pickle(args.source)

    thresh = args.threshold * len(db)

    print("Detecting columns...")

    cols_to_drop = []
    for col in db.columns:
        _, col_dist = np.unique(db[col], return_counts=True)
        if np.any(col_dist >= thresh):
            cols_to_drop.append(col)

    print("Dropping columns", cols_to_drop)

    db = db.drop(cols_to_drop, axis=1)

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
