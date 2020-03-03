import numpy as np
import pandas as pd
import argparse as ap
import models.xgb as xgb
import models.data_util


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('--out_model', type=str,
        default="data/xgb_model.bin",
        help="Path to the location to store the trained model.")
    parser.add_argument('data', type=str,
        help="Path to data file, in Pickle format.")
    parser.add_argument('--train_test_split', type=float,
                        default=0.8,
                        help="The proportion of the dataset to use for training.")
    
    args = parser.parse_args()

    print("Loading training data...")

    # Use helper function to convert everything to numpy etc:
    # NOTE: we are temporarily selecting a subset of the columns
    # to avoid having to convert non-binary categorical variables
    # to a one-hot encoding.
    x, y = models.data_util.load_raw_data(args.data, cols=['lat',
        'long', 'namechanged', 'namechanged2', 'imd', 'imdu', 'nSIC',
        'oac1', 'CompanyNameCountNum', 'CompanyNameCountX',
        'CompanyNameLen', 'CompanyNameWordLen'])

    # Split data into train/test
    N = len(x)
    assert(N == len(y))
    N_train = int(N * args.train_test_split)

    x_train, y_train = x[:N_train], y[:N_train]
    x_test, y_test = x[N_train:], y[N_train:]

    print("Training... [ on a dataset of size", N_train, "]")

    # Train the model:
    model = xgb.from_training_data(x_train, y_train)

    if N_train < N:
        print("Evaluating model... [ on a dataset of size",
              N - N_train, "]")

        y_pred = model.predict(x_test)

        print("Results:",
            np.count_nonzero(y_test == y_pred),
            "correct predictions, out of",
            len(y_test))

    print("Saving model...")

    # Save model and then exit:
    model.save_model(args.out_model)

    print("Done.")
