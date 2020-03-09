# adversarial-ml
2nd Year Group Project - Team 10 Project B

This project aims to investigate black-box attacks on credit-scoring models, and suggest ways of defending against them.

## Installation

After cloning the repository, change the working directory to the root directory of the project and type the
command `pip install requirements.txt` to install all of the Python packages needed for this project.

Finally, make sure to place the data sent to us by TradeTeq into the data folder.

## Navigation

- The `notebooks` folder contains Jupyter notebooks, which contain some nice visualisations.
- The `scripts` folder contains executable scripts, for example model training & testing, data cleaning, etc. There is a separate `feature` subfolder for data cleaning.
- The `models` folder contains code for creating models (intended to be imported, not directly executed.)

## Training and running the XGBoost model

In order to train and run the XGBoost model, follow these steps:
- CD to the root repo directory,
- Place the training data in the "data" folder,
- Execute: `python <script_filename> <training_data> <training_data_clean>` to clean the dataset, for every script in the `scripts/feature` folder.
- Execute: `python scripts/train_xgb.py <training_data_clean>` to train the model and save the model weights.
- Execute: `python scripts/run_xgb.py <training_data_clean>` to run the model.
