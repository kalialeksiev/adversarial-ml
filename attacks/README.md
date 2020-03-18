# attacks folder readme

This folder contains code to be imported for attacks. Herein is a description of some of the attacks:

## The RBM attack

The RBM (restricted Boltzmann machine) attack tries to fool the model by removing information about a company only. Of course, this attack is quite restricted (it can *only* try removing data about a company to fool the model). This attack is more of a sanity-check to ensure that the model isn't gullible enough to fall for something like this.

The restricted Boltzmann machine acts as an unsupervised model of the input data (where each feature is either (i) already a boolean flag or (ii) is turned into a boolean flag, which is the indicator of: the event that the data for that column is present).
