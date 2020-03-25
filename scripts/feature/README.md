# Feature folder readme

This folder is for data-cleaning scripts. Most of these scripts use information from the `models.feature_util` module.

## Summary:

- `drop_cols` : remove several columns that we do not want to train a model on,
- `onehot` : turn categorical columns into one-hot ones,
- `clean_acc_fields` : clean the accounting fields (the ones of the form `FieldN` for some number N) by transforming to log-space,
- `latlong_pca` : use Kernelised Principal Component Analysis to project the latitude and longitude of the companies into a different space, which is easier to model,
- `clean_dates` : standardise all date-like fields so they are represented by a single decimal number, by subtracting the current date from them, and also transforming them to log-space and removing date fields which highly correlate with one another,
- `clean_imd` : clean the `imd` and `imdu` fields (an "index of mass deprivation"), by making them normally distributed,
- `drop_inbalances` : remove columns which take one particular value 99.9% of the time (the reasoning being that, the 0.1% of the time where we don't observe that value is a special case on which we don't have enough data anyway). It is ideal to run this script *LAST* because it will remove such columns from, e.g. the onehot script, which creates a lot of new columns,
- `impute` : fill in all missing values (NaNs and Nones). Warning: this does not create corresponding indicator columns, so information may be lost if one is not careful,
- `drop_correlations` : drop any column which highly correlates with other columns (the idea being that one of the columns is redundant),
- `std_scale` : convert all non-binary columns to have mean 0 variance 1.

## Recommended order:

The first six scripts in the list above can be done in any order, really. The `drop_inbalances` and `drop_correlations` scripts should be done next, as they will operate on the modified columns, for example operating on the results from `onehot`. Finally, do the `impute` and `std_scale` scripts; doing these last reduces unnecessary work in imputing columns that would be dropped anyway.

For the neural network model, it is recommended to run all of these scripts (and doing so results in the possibility of obtaining around 0.9 AUROC). For the XGBoost model, it will work fine without the `impute` script (because it naturally handles NaNs anyway) but you may still want to use it - still use all other scripts for XGBoost.
