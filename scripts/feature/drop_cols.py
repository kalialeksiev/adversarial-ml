import pandas as pd
import argparse as ap
import pickle
from models.data_util import get_col_matcher
from models.feature_util import drop_cols



"""This script will drop some columns from a dataset, for
cleaning. The dropped columns are either not useful, or
'too good', for example hasGNotice.
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

    pred = get_col_matcher(drop_cols)  # a predicate for dropping columns

    # compute which columns to remove
    cols_to_remove = [col for col in db.columns if pred(col)]

    print("Removing", len(cols_to_remove), "columns...")

    db = db.drop(cols_to_remove, axis=1)

    print("Removed:", cols_to_remove)

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
