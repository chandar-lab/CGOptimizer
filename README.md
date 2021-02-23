# Critical Gradient Optimizers

Critical Gradient Optimizers from the "Memory Augmented Optimizers" project and paper, stripped down to just the necessary components to integrate the optimizers into your code.

## Installation

Clone this repository into your project folder. You can import the optimizers as you would any PyTorch optimizer. There are no requirements to run this other than PyTorch.

## Importing and Running

When installed in your project folder, import the optimizers to your training script as needed:

```
from CGOptimizer.optim import SGD_C, RMSprop_C, Adam_C
```

You can then replace any PyTorch optimizer in your script with their `_C` counterpart. Note that currently only Critical-Gradient variants of Adam, RMSprop, and SGD (with optional momentum but NOT Nesterov) are implemented.

Here is a sample replacement:

```
optimizer = Adam(model.parameters(), lr=0.001)
```

becomes

```
optimizer = Adam_C(model.parameters(), lr=0.001, **kwargs)
```

## Optimizer Usage and Tuning

The Critical Gradient variants use all the same hyperparameters as their vanilla counterparts, so you may not need to perform any additional tuning.

The `_C` optimizers have two additional hyperparameters compared to the vanilla version: `topC` which indicates how many critical gradients to keep and`decay` which indicates how much the norms of critical gradients are decayed each step. These are keyword arguments with default values which we observed to work well. For additional performance, these can be tuned.

The `_C` variants perform best using either the same best learning rate as its vanilla counterpart, or 1/10 that learning rate. It is recommended you run both learning rates to compare.

