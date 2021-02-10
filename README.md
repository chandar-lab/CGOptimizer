# Offline-Gradient-Optimization

## Data download

Separate download only necessary for certain datasets. Please see respective folders for instructions on acquiring data.

## Training the models

This code was designed to run on a SLURM cluster, and is optimized to run as an array of jobs with several workers. All hyperparameters can be set in the `PARAM_GRID` variable of the training script.

Uses WandB to log train/valid/test data as well as hyperparameter configuration. Additional steps (e.g. using `dryrun`) may be needed to run depending on the system.

Models and training scripts are segmented by architecture and/or dataset.

To run code from the home folder,
```
python <director>/train.py --data_path <data-directory> --results_path <wandb-result-directory>
```

The `--data_path` and `--results_path` arguments are optional, and will use default locations if not specified. Additional arguments can be passed to affect the model/training. These are dependent on the dataset/model.

Currently you can choose the following models: 

```
FC-NeuralNet, LogisticRegression, ConvNet for MNIST dataset.

LSTM for PTB and WikiText

RESNET, ConvNet for CIFAR

LSTMEncoder, Infersent, and ConvNetEncoder for SNLI

RoBERTa-Base and Bi-LSTM for MultiWoZ
```