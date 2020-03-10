import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import KernelPCA
import argparse as ap
import pickle


"""
This script uses kernelised-PCA (principal component analysis)
to project the latitude and longitude of each position into a
"nicer" space. The kernel we use is the RBF (radial basis function)
kernel.
Warning: this is computationally expensive! The kernel-fitting
process is O(n^2) memory, so it's completely infeasible to fit
with the whole dataset, thus we must fit with a fraction. But even
once we have obtained the PCA transform, transforming the rest of
the dataset is still very intensive!
"""


if __name__ == "__main__":
    
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('source', type=str,
        help="Raw input data (PKL file).")
    parser.add_argument('dest', type=str,
        help="Output location (PKL file).")
    parser.add_argument('--fitting_frac', type=float,
        default=3000.0/600000.0,
        help="The fraction of the data to fit the PCA to.")

    args = parser.parse_args()

    print("Loading input...")

    db = pd.read_pickle(args.source)

    print("Projecting 'lat' and 'long'...")
    
    # Create the kernel PCA object (project into 2 dimensions)
    transformer = KernelPCA(n_components=2, kernel='rbf',
                            copy_X=False)

    print("[Fitting transform...]")

    # only fit the kernel PCA to a subset of the data, because
    # this fitting process is O(n^2) memory, and would require
    # about 2 TB of RAM!
    # Thus only fit to a certain fraction of the dataset:
    transformer.fit_transform(
        db[['lat', 'long']].sample(frac=args.fitting_frac)
    )

    print("[Applying transform to entire dataset...]")

    db[['lat', 'long']] = db[['lat', 'long']].apply(lambda x: np.squeeze(transformer.transform(x.to_numpy().reshape((1, 2)))),
                                                    axis=1,
                                                    result_type='expand')

    # rename columns (rather than creating a new pair of columns)
    # to be more memory efficient
    db = db.rename(columns={'lat' : 'pos1',
                            'long' : 'pos2'})

    print("Done. Saving...")

    pickle.dump(db, open(args.dest, "wb"))
