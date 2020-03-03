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
    parser.add_argument('--data', type=str,
        help="Path to data file, in Pickle format.")
    
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

    print("Training...")

    # Train the model:
    model = xgb.from_training_data(x, y)

    print("Saving results...")

    # Save model and then exit:
    model.save_model(args.out_model)

    print("Done.")
