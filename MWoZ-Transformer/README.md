###Dataset

Download [dataset](https://github.com/ppartha03/Dialogue-Probe-Tasks-Public) and follow the steps in the repo to construct `MultiWoZ_test, MultiWoZ_train and MultiWoZ_val.csv` and run `python transformer_preprocess.py` to generate the necessary data files to train RoBERTa model on MultiWoZ dataset.

###Train and Test

Then, run `python dialog_transformer.py` to train and test the model.
