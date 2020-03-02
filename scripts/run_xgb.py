import numpy as np
import pandas as pd
import argparse as ap
import models.xgb as xgb
import models.data_util


if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ap.ArgumentParser()

    parser.add_argument('--model', type=str,
        default="data/xgb_model.bin",
        help="Path to the location from which to obtain the trained model.")
    parser.add_argument('--data', type=str,
        default="data/Co_600K_Jul2019_6M.pkl",
        help="Path to data file to predict on, in Pickle format.")
    
    args = parser.parse_args()

    print("Loading data...")

    # Use helper function to convert everything to numpy etc:
    # NOTE: we are temporarily selecting a subset of the columns
    # to avoid having to convert non-binary categorical variables
    # to a one-hot encoding.
    x, y_true = models.data_util.load_raw_data(args.data, cols=['lat',
        'long', 'namechanged', 'namechanged2', 'imd', 'imdu', 'nSIC',
        'oac1', 'CompanyNameCountNum', 'CompanyNameCountX',
        'CompanyNameLen', 'CompanyNameWordLen'])

    print("Loading model...")
    
    model = xgb.from_file(args.model)

    print("Predicting...")

    y_pred = model.predict(x)

    print("Results:",
        np.count_nonzero(y_true == y_pred),
        "correct predictions, out of",
        len(y_true))
