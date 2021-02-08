from Model.Convnets import LogisticRegression
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C

from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product
import os
import numpy as np
import random

import torch
import torch.nn as nn
mem = Memory("./mycache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb

os.environ["WANDB_API_KEY"] = '90b23c86b7e5108683b793009567e676b1f93888'
os.environ["WANDB_MODE"] = "dryrun"


#@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def HyperEvaluate(config):

    torch.manual_seed(config['seed'])

    BATCH_SIZE = 25
    N_EPOCHS = 25

    wandb.init(project="convergence", reinit = True)

    wandb.config.update(config)

    if config['dataset'] == 'covtype' :
        X, y = get_data("./covtype.bz2")
        X = X.toarray()
        X = X[np.random.randint(X.shape[0], size=5000), :]
        y = y[np.random.randint(y.shape[0], size=5000)]
    elif config['dataset'] == 'rcv1':
        X, y = get_data("./rcv1_train.binary.bz2")
        X = X.toarray()
        X = X[np.random.randint(X.shape[0], size=5000), :]
        y = 0.5* y[np.random.randint(y.shape[0], size=5000)] +0.5


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Preprocessing step

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).long()
    y_test = torch.from_numpy(y_test.astype(np.float32)).long()



    model = LogisticRegression(X.shape[1], int(max(y_train)+1) )
    model = model.to(device)

    print(config['model'],':',sum(p.numel() for p in model.parameters() if p.requires_grad))

    weight_decay = 0
    if config['l2']:
        weight_decay = 0.01

    if config['optim'] == 'SGD':
        optimizer = SGD(model.parameters(),lr = config['lr'], weight_decay=weight_decay)
    elif config['optim'] == 'SGDM':
        optimizer = SGD(model.parameters(),lr = config['lr'], momentum = 0.9, weight_decay=weight_decay)
    elif config['optim'] == 'SGD_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], aggr = config['aggr'], weight_decay=weight_decay)
    elif config['optim'] == 'SGDM_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], aggr = config['aggr'], weight_decay=weight_decay)
    elif config['optim'] == 'Adam_C':
        optimizer = Adam_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], aggr = config['aggr'], weight_decay=weight_decay)
    elif config['optim'] == 'Adam':
        optimizer = Adam(model.parameters(), lr = config['lr'], weight_decay=weight_decay)
    elif config['optim'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr = config['lr'], weight_decay=weight_decay)
    elif config['optim'] == 'RMSprop_C':
        optimizer = RMSprop_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], aggr = config['aggr'], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)


    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        for src, trg in data_iter(BATCH_SIZE, X_train, y_train):

            src, trg = src.to(device), trg.to(device)
            output, hidden = model(src)
            loss = criterion(output, trg)


            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            scaled_loss = epoch_loss/y_train.shape[0]

        wandb.log({"Train Loss": epoch_loss, 'Train Loss Scaled': scaled_loss})


PARAM_GRID = list(product(
    ['LR'],             # model
    [100, 101, 102, 103, 104], # seeds
    ['rcv1'],          # dataset
    ['SGD_C', 'Adam_C', 'SGDM_C', 'RMSprop_C'], # optimizer
    [0.01, 0.001, 0.0001, 0.00001],  # lr
    [0.7],  # decay
    [10],  # topC
    ['sum', 'mean'],         # sum
    [1.0],               # kappa
    [True]      # l2
))

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

    params = PARAM_GRID[param_ix]


    m, s, d, o, l, dec, t, ch, k, ts = params

    config = {}
    config['model'] = m
    config['seed'] = s
    config['lr'] = l
    config['dataset'] =d
    config['optim'] = o
    config['decay'] = dec
    config['aggr'] = ch
    config['topC'] = t
    config['kappa'] = k
    config['l2'] = ts

    HyperEvaluate(config)

