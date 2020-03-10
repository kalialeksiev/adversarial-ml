import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import KernelPCA
import argparse as ap
import pickle
from models.feature_util import (accounting_field_nums,
                                 bad_accounting_field_nums)


"""
This script will clean all of the accounting fields in the data
(the fields of the form FieldN for some number N).
This cleaning process includes:
- Removing the redundancy with hasFN by setting FieldN to be None
  when hasFN==0
- Dropping accounting fields which don't have a corresponding hasFN
  (this is perhaps a bad thing, however these columns don't contain
  strictly numerical data).
- Optionally applies a logarithm to accounting fields (makes the values
  much more nicely spread out).
"""


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")
    parser.add_argument('--log_data', type=bool,
        default=True,
        help="Transform accounting fields by taking logs?")

    args = parser.parse_args()

    print("Loading input...")

    db = pd.read_pickle(args.source)

    print("Cleaning accounting fields...")

    # our aim here is to remove the redundancy with the
    # fields and the accompanying 'hasFieldN' by letting
    # None represent an empty value
    for n in accounting_field_nums:
        field_name = 'Field' + str(n)
        has_field_name = 'hasF' + str(n)

        if has_field_name in db.columns:
            # do a vectorized conditional assignment to None-out the
            # missing fields
            db.loc[db[has_field_name] == 0, field_name] = None

            # for safety, convert NaNs to Nones
            db.loc[db[field_name] == np.NaN] = None

            # drop the excess column:
            db = db.drop(has_field_name, axis=1)

        # finally, convert to log scale
        if args.log_data:
            db[field_name] = np.log(db[field_name].to_numpy() + 1.0e-6)
    
    # drop bad accounting fields:
    for n in bad_accounting_field_nums:
        db = db.drop('Field' + str(n), axis=1)

    print("Done. Saving...")

    pickle.dump(db, open(args.dest, "wb"))
