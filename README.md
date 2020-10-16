# Offline-Gradient-Optimization

# Download and setting up the training directory

## Data download

Download only the PTB data. `torchvision.datasets` downloads the CIFAR100 and MNIST.

After cloning the repository.

```
cd CriticalGradientOptimization

mkdir Dataset

cd Dataset

wget https://www.dropbox.com/s/rbuoo026sb276dk/ptb.zip

unzip ptb.zip

```

Make sure the project root has the following folders :

```
cifar\
Dataset\
Model\
Utils\
utilscifar\
word_language_model\
```

Make a copy of the `Results` directory and delete it before running the code.


# Training the models

The code uses ray to manage the experiments. So, set all the hyperparameters within the train script, `mnist.py`, `ptb.py`, and `cifar.py`.

To run the set code,
```

`python mnist.py`

`python cifar.py`

`python ptb.py`

```

Currently you can choose the following models: 

```
FC-NeuralNet, LogisticRegression, ConvNet for MNIST dataset.

LSTM for PTB

RESNET, ALEXNET, VGG, DENSENET, WRN, PRERESNET, RESNEXT for CIFAR100 dataset
```
The results get logged in `Results\` (which gets created during the experiment).

The logs get appended to the previous logs. Don't forget to clear it if you are rerunning with the same parameters.
