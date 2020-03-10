import pandas as pd
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

    print("Saving resulting database...")

    pickle.dump(db, open(args.dest, "wb"))
