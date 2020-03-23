import pandas as pd
import argparse as ap
import pickle
from sklearn.preprocessing import StandardScaler



"""This script will scale all non-binary columns to
have mean 0 variance 1.
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

    print("Scaling...")

    nonbinary_cols = []
    for col in db.columns:
        if db[col].nunique() > 2:
            nonbinary_cols.append(col)
    
    std = StandardScaler()

    db[nonbinary_cols] = std.fit_transform(db[nonbinary_cols])

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
