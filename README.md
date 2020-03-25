# adversarial-ml
2nd Year Group Project - Team 10 Project B

This project aims to investigate black-box attacks on credit-scoring models, and suggest ways of defending against them.

## Installation

After cloning the repository, change the working directory to the root directory of the project and type the command `pip install -r requirements.txt` to install all of the Python packages needed for this project. Next, go to `https://www.graphviz.org/` and install the GraphViz libary. Add the package's `bin` folder to your PATH.

Finally, make sure to place the data sent to us by TradeTeq into the data folder. You should then proceed by cleaning the data, by running the scripts in the `scripts/feature/` folder. Then you should train the neural network and/or XGBoost models using the relevant training scripts in the `scripts/` folder. Then you will have saved model weights available to attack the models.

## Navigation

- The `notebooks` folder contains Jupyter notebooks, which contain some nice visualisations.
- The `scripts` folder contains executable scripts, for example model training & testing, data cleaning, etc. There is a separate `feature` subfolder for data cleaning.
- The `models` folder contains code for creating models (intended to be imported, not directly executed.)

## A note on running scripts

When we run a command like `python scripts/feature/clean_acc_fields.py <args>`, depending on the Python settings, either the current working directory (the root of the repo) will be added to the path to search for other modules, or the script's directory `scripts/feature` will be added. In order to get imports between files in the repo working, it is important that the root directory of the repo is added to the path to search for modules.

If, when executing a script like the above, you are getting import errors when trying to import other models from the repo, you can use an alternative command for running scripts, which will always add the working directory to the path: `python -m scripts.feature.clean_acc_fields <args>`. This "runs the script as a package instead of as a script" - see the documentation for more information on the differences between these two commands.

## Data cleaning

- CD to the root directory
- For each script in the `scripts/feature/` directory, run it using `python -m <script_filename> <training_data> <training_data_clean>` (note that some scripts require additional arguments, too.)
- **Note:** if you are receiving import errors about other files in the project (e.g. `ImportError: cannot find package 'models'`) see the above.

## Training and running the XGBoost model

In order to train and run the XGBoost model, follow these steps:
- Ensure the cleaned dataset is somewhere, e.g. in the `data` folder.
- Execute: `python -m scripts.train_xgb <training_data_clean>` to train the model and save the model weights. This will tell you the AUROC achieved.
- You can also execute: `python -m scripts.run_xgb <training_data_clean>` to run the model, however this will run the model on the whole dataset (i.e. including the training set) so expect the AUROC to be slightly higher than expected!

## Training and running the neural network model

In order to train and run the neural network model, follow these steps:
- Ensure the cleaned dataset is somewhere, e.g. in the `data` folder.
- Execute: `python -m scripts.train_nn <training_data_clean>` to train the model and save the model weights. This will tell you the AUROC achieved. You can tweak with many of the hyperparameters, including the number of training epochs. See the command line argument options of the script for help.

