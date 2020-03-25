import pandas as pd
import numpy as np
import argparse as ap
import pickle



"""This script will drop features which correlate highly
with another feature (which is not also being dropped).
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")
    parser.add_argument('--threshold', type=float,
        default=0.98,
        help="The minimum proportion required to drop.")

    args = parser.parse_args()

    print("Loading input...")

    db = pd.read_pickle(args.source)
    cols = list(db.columns)

    print("Cleaning...")

    # now compute correlation matrix

    corr_mat = db.dropna().corr()

    # remove columns which correlate highly with another column
    drop_cols = []
    for i, col_i in enumerate(cols):
        for j, col_k in enumerate(cols[i+1:]):
            k = j + i + 1
            if (abs(corr_mat[col_i][col_k]) > args.threshold
                and not col_k in drop_cols):
                drop_cols.append(col_k)
    print("Dropping columns due to high correlation:", drop_cols)

    db = db.drop(drop_cols, axis=1)  # safe because correlation is transitive and symmetric

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
