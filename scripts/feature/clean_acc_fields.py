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

    print("Cleaning accounting fields...")

    # our aim here is to remove the redundancy with the
    # fields and the accompanying 'hasFieldN' by letting
    # None represent an empty value
    for n in accounting_field_nums:
        # do a vectorized conditional assignment to None-out the
        # missing fields
        db.loc[db['hasF' + str(n)] == 0, 'Field' + str(n)] = None

        # for safety, convert NaNs to Nones
        db.loc[db['Field' + str(n)] == np.NaN] = None

        # finally, drop the excess column:
        db = db.drop('hasF' + str(n), axis=1)
    
    # drop bad accounting fields:
    for n in bad_accounting_field_nums:
        db = db.drop('Field' + str(n), axis=1)

    print("Done. Saving...")

    pickle.dump(db, open(args.dest, "wb"))
