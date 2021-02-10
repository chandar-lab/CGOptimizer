# Download and set up the training directory

`torchvision.datasets` downloads the MNIST data automatically. Make sure data is downloaded ahead of time if running on an offline system.

After cloning the repository, create a Dataset directory in your `mnist` directory. Code will download data and search this directory by default.

```
cd CriticalGradientOptimization

cd mnist

mkdir Dataset

cd Dataset
```