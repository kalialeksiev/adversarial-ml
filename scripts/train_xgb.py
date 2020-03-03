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
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help="Minimum probability to be considered a 'yes' example.")

    args = parser.parse_args()

    print("Loading training data...")

    # Use helper function to convert everything to numpy etc:
    # NOTE: we are temporarily selecting a subset of the columns
    # to avoid having to convert non-binary categorical variables
    # to a one-hot encoding.
    x, y = models.data_util.load_raw_data(args.data, cols=['lat',
        'long', 'namechanged', 'namechanged2', 'nSIC',
        'CompanyNameCountNum', 'CompanyNameCountX',
        'CompanyNameLen', 'CompanyNameWordLen', 'MortgagesNumMortCharges',
        'MortgagesNumMortOutstanding', 'MortgagesNumMortPartSatisfied',
        'MortgagesNumMortSatisfied', 'SIC1', 'SIC2', 'SIC3'],
        shuffle=True)

    # TEMP: make the dataset more balanced by reducing some of the
    # (abundance of) negative examples.
    idxs = np.nonzero(y == 1)
    x = np.concatenate((x[:4000], x[idxs]), axis=0)
    y = np.concatenate((y[:4000], y[idxs]), axis=0)
    
    # Now shuffle the dataset again:
    idxs = np.random.shuffle(np.arange(len(x)))
    x = np.squeeze(x[idxs], axis=0)
    y = np.squeeze(y[idxs], axis=0)

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
        # convert probabilities to predictions via thresholding:
        y_pred = models.data_util.threshold(y_pred, args.threshold)

        # compute accuracy separately on positive and negative examples
        true_neg, true_pos = models.data_util.get_accuracy(
            y_test, y_pred
        )

        print("Results:", "True negative rate =",
              true_neg, "True positive rate =", true_pos,
              "ROC curve =", roc)

    print("Saving model...")

    # Save model and then exit:
    model.save_model(args.out_model)

    print("Done.")
