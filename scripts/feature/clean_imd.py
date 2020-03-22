import pandas as pd
import argparse as ap
import pickle
from sklearn.preprocessing import quantile_transform



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

    # compute the difference between the two fields (I thought
    # this would be useful because they are both quite similar
    # fields, so no point having both.)
    db['imdu'] = db['imd'] - db['imdu']

    # rename the column, as it now represents the difference of imd and imdu:
    db = db.rename(columns={'imdu' : 'imdu_diff'})

    # transform these features (which look like uniform distributions)
    # into normally-distributed features
    db[['imd', 'imdu_diff']] = quantile_transform(db[['imd', 'imdu_diff']],
                                                  output_distribution='normal')

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
