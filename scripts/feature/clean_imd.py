import pandas as pd
import argparse as ap
import pickle



"""This script standardizes the 'imd' and 'imdu' columns
in the dataset.
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

    print("Cleaning...")

    db['imdu'] = db['imd'] - db['imdu']

    # standardise:
    db['imdu'] = (db['imdu'] - db['imdu'].mean()) / db['imdu'].std()

    # rename:
    db = db.rename(columns={'imdu' : 'imdu_diff'})

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
