# Download and set up the training directory

`torchvision.datasets` downloads the CIFAR10/100 data automatically. Make sure data is downloaded ahead of time if running on an offline system.

After cloning the repository, create a Dataset directory in your `cifar` directory. Code will download data and search this directory by default.

```
cd CriticalGradientOptimization

cd cifar

mkdir Dataset

cd Dataset
```