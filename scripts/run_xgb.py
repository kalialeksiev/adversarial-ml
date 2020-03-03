import numpy as np
import pandas as pd
import sklearn as skl
import argparse as ap
import models.xgb as xgb
import models.data_util


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('--model', type=str,
        default="data/xgb_model.bin",
        help="Path to the location from which to obtain the trained model.")
    parser.add_argument('data', type=str,
        help="Path to data file to predict on, in Pickle format.")
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help="Minimum probability to be considered a 'yes' example.")
    
    args = parser.parse_args()

    print("Loading data...")

    # Use helper function to convert everything to numpy etc:
    # NOTE: we are temporarily selecting a subset of the columns
    # to avoid having to convert non-binary categorical variables
    # to a one-hot encoding.
    x, y_true = models.data_util.load_raw_data(args.data, cols=['lat',
        'long', 'namechanged', 'namechanged2', 'nSIC',
        'CompanyNameCountNum', 'CompanyNameCountX',
        'CompanyNameLen', 'CompanyNameWordLen', 'MortgagesNumMortCharges',
        'MortgagesNumMortOutstanding', 'MortgagesNumMortPartSatisfied',
        'MortgagesNumMortSatisfied', 'SIC1', 'SIC2', 'SIC3',
        'country', 'cty', 'eAccountsAccountCategory', 'eCompanyCategory'])

    print("Loading model...")
    
    model = xgb.from_file(args.model)

    print("Predicting...")

    # predict probabilities of positive class:
    y_pred = model.predict_proba(x).T[1].T
    # get the area under the ROC graph:
    fpr, tpr, _ = skl.metrics.roc_curve(y_true, y_pred)
    roc = skl.metrics.auc(fpr, tpr)
    # convert probabilities to predictions via thresholding:
    y_pred = models.data_util.threshold(y_pred, args.threshold)

    # compute accuracy separately on positive and negative examples
    true_neg, true_pos = models.data_util.get_accuracy(
        y_true, y_pred
    )

    print("Results:", "True negative rate =",
          true_neg, "True positive rate =", true_pos,
          "ROC curve =", roc)
