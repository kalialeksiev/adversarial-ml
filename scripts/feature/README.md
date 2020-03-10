# Feature folder readme

This folder is for data-cleaning scripts. Most of these scripts use information from the `models.feature_util` module.

## Summary:

- `drop_cols` : remove several columns that we do not want to train a model on,
- `onehot` : turn categorical columns into one-hot ones,
- `clean_acc_fields` : clean the accounting fields (the ones of the form `FieldN` for some number N) by removing the redundancy with the columns `FieldN` and `hasFN`, and also transforming to log-space,
- `latlong_pca` : use Kernelised Principal Component Analysis to project the latitude and longitude of the companies into a different space, which is easier to model.
- `clean_dates` : standardise all date-like fields so they are represented by a single decimal number, by subtracting the current date from them.
