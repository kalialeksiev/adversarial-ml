# Feature folder readme

This folder is for data-cleaning scripts. Most of these scripts use information from the `models.feature_util` module.

## Summary:

- `drop_cols` : remove several columns that we do not want to train a model on,
- `onehot` : turn categorical columns into one-hot ones,
- `clean_acc_fields` : clean the accounting fields (the ones of the form `FieldN` for some number N) by transforming to log-space,
- `latlong_pca` : use Kernelised Principal Component Analysis to project the latitude and longitude of the companies into a different space, which is easier to model,
- `clean_dates` : standardise all date-like fields so they are represented by a single decimal number, by subtracting the current date from them,
- `clean_imd` : clean the `imd` and `imdu` fields (an "index of mass deprivation"),
- `drop_inbalances` : remove columns which take only two values, and where one of those values occurs 99.9% of the time (the reasoning being that, the 0.1% of the time where we don't observe that value is a special case on which we don't have enough data anyway). It is ideal to run this script *LAST* because it will remove such columns from, e.g. the onehot script, which creates a lot of new columns.
