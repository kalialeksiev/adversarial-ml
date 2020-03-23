import pandas as pd
import numpy as np
import argparse as ap
import pickle
from datetime import datetime
from models.data_util import get_col_matcher, date_to_decimal
from models.feature_util import date_cols



"""This script will clean all date/time-related fields
for the model to use.
Running this script requires inputting the date on which
the data was sampled (this is to subtract from the dates
in the columns, to make the dates relative.)
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")
    parser.add_argument('date', type=str,
        help="The data sampling date in DD/MM/YYYY format.")
    parser.add_argument('--drop_threshold', type=float,
                        default=0.98,
                        help="If two variables correlate more than"
                             " this value, one of them will be dropped.")

    args = parser.parse_args()

    curdate = datetime.strptime(args.date, '%d/%m/%Y')

    # convert date to a decimal which represents the year plus the
    # fraction of the year traversed through:
    curdate = date_to_decimal(curdate)

    # now get the date as a year fraction

    print("Loading input...")

    db = pd.read_pickle(args.source)

    pred = get_col_matcher(date_cols)  # a predicate for date-based columns

    # compute which columns are dates
    cols = [col for col in db.columns if pred(col)]

    print("Cleaning...")

    # subtract current date from all given dates:
    db[cols] -= curdate

    # now compute correlation matrix

    corr_mat = db[cols].dropna().corr()

    # remove columns which correlate highly with another column
    drop_cols = []
    for i, col_i in enumerate(cols):
        for j, col_k in enumerate(cols[i+1:]):
            k = j + i + 1
            if abs(corr_mat[col_i][col_k]) > args.drop_threshold:
                drop_cols.append(col_k)
    print("Dropping columns due to high correlation:", drop_cols)

    db = db.drop(drop_cols, axis=1)  # safe because correlation is transitive and symmetric

    print("Transforming to log space...")

    # what date columns are we left with?
    rest_cols = list(set(cols) - set(drop_cols))

    # transform to log space (do this AFTER correlation computations,
    # as the log obfuscates linear relationships.)
    db[rest_cols] = np.log(np.abs(db[rest_cols] + 1.0e-6))

    print("Creating presence indicators...")

    # create a set of indicator columns to demonstrate the presence/absence
    # of each date input:
    ind_cols = ["has_" + col for col in rest_cols]
    db[ind_cols] = db[rest_cols].notna()

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
