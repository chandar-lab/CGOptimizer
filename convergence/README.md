# Download and set up the training directory

A separate download is necessary for the covtype/rcv1 datasets.

After cloning the repository, create a Dataset directory in your `convergence` directory. Code will search this directory by default.

```
cd CriticalGradientOptimization

cd convergence

mkdir Dataset

cd Dataset
```

For covtype:

```
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.bz2
```

For rcv1 (training only):

```
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
```