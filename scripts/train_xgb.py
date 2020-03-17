import numpy as np
import pandas as pd
import sklearn as skl
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
    x, y = models.data_util.load_raw_data(args.data, shuffle=True)

    # Split data into train/test
    N = len(x)
    assert(N == len(y))
    N_train = int(N * args.train_test_split)

    x_train, y_train = x[:N_train], y[:N_train]
    x_test, y_test = x[N_train:], y[N_train:]

    print("Training... [ on a dataset of size", N_train, "]")

    # Train the model:
    model = xgb.from_training_data(x_train, y_train)

    if N_train < N:  # if we need to perform evaluation...
        print("Evaluating model... [ on a dataset of size",
              N - N_train, "]")

        # predict probabilities of positive class:
        y_pred = model.predict_proba(x_test).T[1].T
        # get the area under the ROC graph:
        fpr, tpr, _ = skl.metrics.roc_curve(y_test, y_pred)
        roc = skl.metrics.auc(fpr, tpr)

        print("Results:",
              "Area under ROC curve =", roc)

    print("Saving model...")

    # Save model and then exit:
    model.save_model(args.out_model)

    print("Done.")
