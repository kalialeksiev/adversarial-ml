import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import KernelPCA
import argparse as ap
import pickle


"""
This is a script useful for producing a cleaner version of the
original data.

More specifically, this script does several things:

- Uses Kernelised PCA (principal component analysis) using RBF (radial
  basis function) kernel to transform the lat/long coordinates of the
  company to an easier space,
- Converts some categorical variables to one-hot,
- Handles missing values etc.
"""


raw_cols = [  # columns we will consider, as given in the raw data file
    'isfailed',  # the target
    'lat', 'long', 'namechanged', 'namechanged2', 'nSIC',
    'MortgagesNumMortCharges', 'MortgagesNumMortOutstanding',
    'MortgagesNumMortPartSatisfied', 'MortgagesNumMortSatisfied',
    'SIC1', 'SIC2', 'SIC3', 'AccountsAccountCategory',
    'CompanyCategory',
    'Field1014', 'Field1129', 'Field1522', 'Field1631',
    'Field17', 'Field1865', 'Field1871', 'Field1885',
    'Field1977', 'Field2267', 'Field2298', 'Field2304',
    'Field2316', 'Field2447', 'Field2483', 'Field2497',
    'Field2502', 'Field2506', 'Field2616', 'Field2619',
    'Field2705', 'Field2815', 'Field2816', 'Field282',
    'Field2823', 'Field306', 'Field448', 'Field465',
    'Field474', 'Field477', 'Field487', 'Field489',
    'Field541', 'Field69', 'Field70', 'Field972',
    'hasF1014', 'hasF1129', 'hasF1522', 'hasF1631',
    'hasF17', 'hasF1865', 'hasF1871', 'hasF1885',
    'hasF1977', 'hasF2298', 'hasF2304',
    'hasF2316', 'hasF2447', 'hasF2483', 'hasF2497',
    'hasF2502', 'hasF2506', 'hasF2616', 'hasF2619',
    'hasF2705', 'hasF2815', 'hasF282',
    'hasF306', 'hasF448', 'hasF465',
    'hasF474', 'hasF487', 'hasF489',
    'hasF541', 'hasF69', 'hasF70'
]
# WARNING: the following accounting fields do not have a corresponding "hasF...":
# Field2267, Field2816, Field972, Field477, Field2823
# I have noticed that these columns, and only these columns (I think), contain None
# values, as well as NaN etc.


categorical_cols = [  # which columns should we turn into one-hot?
    'SIC1', 'SIC2', 'SIC3', 'nSIC',
    'AccountsAccountCategory', 'CompanyCategory'
]


accounting_field_nums = [  # obtained from the raw_cols list above (note the warning above, too)
    1014, 1129, 1522, 1631,
    17, 1865, 1871, 1885,
    1977, 2298, 2304,
    2316, 2447, 2483, 2497,
    2502, 2815, 282,
    306, 448, 465,
    474, 487, 489,
    541, 69, 70
]


bad_accounting_field_nums = [  # see warning above
    2267, 2816, 972, 477, 2823
]


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

    print("Projecting 'lat' and 'long'...")
    
    # Create the kernel PCA object (project into 2 dimensions)
    transformer = KernelPCA(n_components=2, kernel='rbf')

    print("[Fitting transform...]")

    # only fit the kernel PCA to a subset of the data, because
    # this fitting process is O(n^2) memory, and would require
    # about 2 TB of RAM!
    # Thus only fit to a certain fraction of the dataset:
    transformer.fit_transform(
        db[['lat', 'long']].sample(frac=3000.0/600000.0)
    )

    print("[Applying transform to entire dataset...]")

    # create new columns
    db['pos1'] = np.NaN
    db['pos2'] = np.NaN

    # now apply the learned transform to the whole dataset:
    # (warning: expensive!)
    db[['pos1', 'pos2']] = transformer.transform(
        db[['lat', 'long']].to_numpy()
    )

    # drop lat and long as they are no longer needed:
    db = db.drop(['lat', 'long'], axis=1)

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

    print("One-hotting categorical columns...")

    db = pd.get_dummies(db, prefix=categorical_cols,
                        columns=categorical_cols)

    print("Done. Saving...")

    pickle.dump(db, open(args.dest, "wb"))
