import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.metrics
import argparse as ap
import models.nn as nn
import models.data_util
from tensorflow.python.keras.utils.np_utils import to_categorical


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('--out_model', type=str,
        default="data/nn_model",
        help="Path to the location to store the trained model.")
    parser.add_argument('data', type=str,
        help="Path to data file, in Pickle format.")
    parser.add_argument('--train_test_split', type=float,
                        default=0.8,
                        help="The proportion of the dataset to use for training.")
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help="Minimum probability to be considered a 'yes' example.")
    parser.add_argument('--epochs', type=int,
                        default=nn.DEFAULT_EPOCHS,
                        help="Minimum probability to be considered a 'yes' example.")

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

    print("Training for", args.epochs, "epochs... [ on a dataset of size", N, "]")

    # Train the model:
    model = nn.from_training_data(x_train, y_train, epochs=args.epochs)

    if N_train < N:  # if we need to perform evaluation...
        print("Evaluating model... [ on a dataset of size",
              N - N_train, "]")

        y_pred = model.predict(x_test).T[1].T
        print(y_pred)
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
    model.save(filepath=args.out_model, overwrite=True)

    print("Done.")
