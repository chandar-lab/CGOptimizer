# from torchtext import data

from Model.Convnets import Cifar10CnnModel, Cifar100CnnModel, ClassDecoder, LogisticRegression, FCLayer

import torch.optim as optim
# from Utils.Eval_metric import getBLEU
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C
from optimizers.optimExperimental import SAGA

from EncoderDecoder import EncoderDecoder
import sys
# sys.path.append('/home/ml/pparth2/anaconda3/lib/python3.7/site-packages')
import torch
import torch.nn as nn
# import numpy as np
import argparse
import os
# import logging
import wandb
# import random
# import math
# import csv
# import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import itertools
from itertools import product

import os.path as osp

from filelock import FileLock

import Model.cifar as models

os.environ["WANDB_API_KEY"] = '90b23c86b7e5108683b793009567e676b1f93888'
os.environ["WANDB_MODE"] = "dryrun"

# commandline arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type = str, default = './Dataset')
parser.add_argument('--results_path', type=str, default = '.')

parser.add_argument('--optimizer',type=str,default='SGD')
parser.add_argument('--model',type=str,default='LR')
parser.add_argument('--dataset',type=str,default='mnist')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--kappa', type=float, default = 1.0)
parser.add_argument('--decay', type=float, default = 0.9)
parser.add_argument('--aggr',type=str,default='average')
parser.add_argument('--topC', type=int, default = 1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.7)

args = parser.parse_args()

data_path = args.data_path
results_path = args.results_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, iterator, optimizer, criterion, clip=10):
    ''' Training loop for the model to train.
    Args:
        model: A EncoderDecoder model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    S = {'yes':0,'no':0}
    id_to_hidden = {}
    # TODO:  convert the hiddenstate to string and save it as text file
    for i, batch in enumerate(iterator):
        stats = None

       # if isinstance(model, Cifar100CnnModel) or isinstance(model, Cifar10CnnModel) or isinstance(model, resnet):
        src = batch[0]
       # else:
       #     src = batch[0].view(-1,784)
        trg = batch[1]
        src, trg = src.to(device), trg.to(device)
        output = model(src)
        loss = criterion(output, trg)
        loss.backward()

        wandb.log({"Iteration Training Loss": loss})

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        if isinstance(optimizer, SAGA):
            optimizer.step(index=i)
        else:
            optimizer.step()
        stats = optimizer.getOfflineStats()
        if stats:
            for k,v in stats.items():
                S[k]+=v
        epoch_loss += loss.item()
    # return the average loss
    return epoch_loss / len(iterator),S

def evaluate(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0
    epoch_correct = 0
    # we don't need to update the model parameters. only forward pass.
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #if isinstance(model, Cifar100CnnModel) or isinstance(model, Cifar10CnnModel):
            src = batch[0]
            #else:
            #    src = batch[0].view(-1,784)
            trg = batch[1]
            src, trg = src.to(device), trg.to(device)
            total += trg.size(0)
            output = model(src)
            loss = criterion(output, trg)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(trg.view_as(pred)).sum()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
    return epoch_loss / len(iterator), 100. * epoch_correct/ total

def HyperEvaluate(config):

    torch.manual_seed(config['seed'])

    if config['optim'] == 'SAGA':
        N_EPOCHS = 25            # number of epochs
        BATCH_SIZE = 1
    else:
        N_EPOCHS = 25           # number of epochs
        BATCH_SIZE = args.batch_size

    if '_C' in config['optim']:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr']) + '_topC_' + str(config['topC']) + '_decay_'+ str(config['decay'])+ '_kappa_' + str(config['kappa']) +'_'+config['aggr']
    else:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr'])

    wandb.init(project="Critical-Gradients-" + config['dataset'], reinit = True)
    wandb.run.name = run_id

    wandb.config.update(config)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if config['dataset'] == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif config['dataset'] == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100

    BATCH_SIZE = args.batch_size

    trainset = dataloader(root='./Dataset', train=True, download=True, transform=transform_train)
    train_iterator = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = dataloader(root='./Dataset', train=False, download=False, transform=transform_test)
    valid_iterator = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    INPUT_DIM = 10

    # encoder

    if config['model'] == 'convnet':
        if config['dataset'] == 'cifar10':
            model = Cifar10CnnModel()
        elif config['dataset'] == 'cifar100':
            model = Cifar100CnnModel()

        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif config['model'] == 'resnet':
        model =  models.__dict__['resnet'](
                    num_classes=num_classes,
                    depth=110,
                    block_name='BasicBlock',
                )

    else:
        print('Error: Model Not There')
        sys.exit(0)
        
    model = model.to(device)

    if config['optim'] == 'SGD':
        optimizer = SGD(model.parameters(),lr = config['lr'])
    elif config['optim'] == 'SGDM':
        optimizer = SGD(model.parameters(),lr = config['lr'], momentum = 0.9)
    elif config['optim'] == 'SGD_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], aggr = config['aggr'])
    elif config['optim'] == 'SGDM_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], aggr = config['aggr'])
    elif config['optim'] == 'Adam_C':
        optimizer = Adam_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], aggr = config['aggr'])
    elif config['optim'] == 'Adam':
        optimizer = Adam(model.parameters(), lr = config['lr'])
    elif config['optim'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr = config['lr'])
    elif config['optim'] == 'RMSprop_C':
        optimizer = RMSprop_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], aggr = config['aggr'])
    criterion = nn.CrossEntropyLoss()

    # loss function calculates the average loss per token
    # passing the <pad> token to ignore_idx argument, we will ignore loss whenever the target token is <pad>


    best_validation_perf = float('-inf')

    for epoch in range(N_EPOCHS):
        valid_perf = 'NA'

        train_loss,offline_stats = train(model, train_iterator, optimizer, criterion )
        valid_loss, valid_perf = evaluate(model, valid_iterator, criterion)

        off = offline_stats['no']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes']*100/(sum([v for v in offline_stats.values()]) + 1e-7)

        optimizer.resetOfflineStats()
        #torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,config['model']+'_'+str(epoch)+'.pt'))
        if valid_perf > best_validation_perf:
           best_validation_perf = valid_perf
          # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,'best_model.pt'))

        wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss, "Val. Accuracy": valid_perf, "offline updates" : off, "online udpates": on})

        #scheduler.step()
    return best_validation_perf


best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}


PARAM_GRID = list(product(
    ['resnet', 'convnet'],             # model
    [100, 101, 102, 103, 104], # seeds
    ['cifar100'],          # dataset
    ['SGD', 'SGDM', 'RMSprop', 'Adam'], # optimizer
    [0.1, 0.01, 0.001, 0.0001],  # lr
    [0],  # decay
    [0],  # topC
    ['none'],         # aggr
    [1.0]               # kappa
))

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

    params = PARAM_GRID[param_ix]


    m, s, d, o, l, dec, t, ch, k = params

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

    accuracy = HyperEvaluate(config)

    print("""We achieve accuracy {:7.2f}% with
        learning_rate: {:.4}
        seed: {}
        Optimizer: {}
      """.format(accuracy, l, s, o))


