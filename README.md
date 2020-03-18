# adversarial-ml
2nd Year Group Project - Team 10 Project B

This project aims to investigate black-box attacks on credit-scoring models, and suggest ways of defending against them.

## Installation

After cloning the repository, change the working directory to the root directory of the project and type the
command `pip install -r requirements.txt` to install all of the Python packages needed for this project. Next,
go to `https://www.graphviz.org/` and install the GraphViz libary. Add the package's `bin` folder to your PATH.

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

## Running the RBM attacks

The RBM (restricted Boltzmann machine) attack tries to fool the model by removing information about a company only. To perform the attack:
- CD to the root repo directory,
- Ensure you have the cleaned data in the "data" folder, and an already trained XGBoost model on that data,
- Execute `python scripts/train_rbm_attack.py <training_data_clean>` to train the RBM model for the attack,
- Execute `python scripts/run_rbm_attack.py <training_data_clean> <num companies to try> <target value>` where you have to provide the number of companies you'd like to try running the attack on (each company will be randomly picked from the dataset) and the target value must be either 0 or 1, where 1 means the attack will try to fool the model into predicting a 1 for `isfailed`, etc.
Of course, this attack is quite restricted (it can *only* try removing data about a company to fool the model). This attack is more of a sanity-check to ensure that the model isn't gullible enough to fall for something like this.
