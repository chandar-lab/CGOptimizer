# Memory Augmented Optimizers

Popular approaches for minimizing loss in data-driven learning often involve an abstraction or an explicit retention of the history of gradients for efficient parameter updates. 
The aggregated history of gradients nudges the parameter updates in the right direction even when the gradients at any given step are not informative. 
Although the history of gradients summarized in meta-parameters or explicitly stored in memory has been shown effective in theory and practice, the question of whether *all* or only a subset of the gradients in the history are sufficient in deciding the parameter updates remains unanswered. 
We propose a class of memory-augmented gradient descent optimizers that retain only the *critical* gradients, as defined by the L2-norm of the gradients, as opposed to the entire history. 
This repository contains these memory-agumented optimizers as well as numerous models to test them on.

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

## Usage

```
from optimizers.optim import SGD_C, Adam_C

# model = any differentiable model built with torch.nn

optimizer = SGD_C(model.parameters(), lr = 1E-2, topC = 5, decay = 0.7)
```
