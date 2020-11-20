# from torchtext import data

from Model.Convnets import ConvNetEncoder, ClassDecoder, LogisticRegression, FCLayer

import torch.optim as optim
# from Utils.Eval_metric import getBLEU
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, SGD_C_Only, SAGA

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
parser.add_argument('--gradsum',type=str,default='average')
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

        if isinstance(model, EncoderDecoder):
            src = batch[0]
        else:
            src = batch[0].view(-1,784)
        trg = batch[1]
        src, trg = src.to(device), trg.to(device)
        output, hidden = model(src)
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
            if isinstance(model, EncoderDecoder):
                src = batch[0]
            else:
                src = batch[0].view(-1,784)
            trg = batch[1]
            src, trg = src.to(device), trg.to(device)
            total += trg.size(0)
            output, hidden = model(src)
            loss = criterion(output, trg)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(trg.view_as(pred)).sum()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
    return epoch_loss / len(iterator), 100. * epoch_correct/ total

def HyperEvaluate(config):
    print(config)


    torch.manual_seed(config['seed'])

    if config['optim'] == 'SAGA':
        N_EPOCHS = 25            # number of epochs
        BATCH_SIZE = 1
    else:
        N_EPOCHS = 25           # number of epochs
        BATCH_SIZE = args.batch_size

    if '_C' in config['optim']:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr']) + '_topC_' + str(config['topC']) + '_decay_'+ str(config['decay'])+ '_kappa_' + str(config['kappa']) +'_'+config['gradsum']
    else:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr'])

    wandb.init(project="Critical-Gradients", reinit = True)
    wandb.run.name = run_id
    wandb.run.save()

    wandb.config.update(config)

    if config['dataset'] == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        with FileLock(os.path.expanduser("~/data.lock")):
            dataset1 = datasets.MNIST(data_path, train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST(data_path, train=False,
                           transform=transform)
        train_iterator = torch.utils.data.DataLoader(dataset1,batch_size=BATCH_SIZE,
                                               shuffle=True)
        valid_iterator = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE,
                                               shuffle=True)
        INPUT_DIM = 10

    # encoder
    if config['model'] == 'LR':
        model = LogisticRegression(784,10)
        model = model.to(device)
        itos_context_id = None
        itos_vocab = None
    elif config['model'] == 'NeuralNet':
        model = FCLayer()
        model = model.to(device)
        itos_context_id = None
        itos_vocab = None
    elif config['model'] == 'convnet':
        enc = ConvNetEncoder().to(device)
        dec = ClassDecoder().to(device)
        model = EncoderDecoder(enc,dec,data='image')
        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'])
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        itos_context_id = None
        itos_vocab = None
    else:
        print('Error: Model Not There')
        sys.exit(0)

    if config['dataset'] == 'mnist':
        if config['optim'] == 'SGD':
            optimizer = SGD(model.parameters(),lr = config['lr'])
        elif config['optim'] == 'SGDM':
            optimizer = SGD(model.parameters(),lr = config['lr'], momentum = 0.9)
        elif config['optim'] == 'SGD_C':
            optimizer = SGD_C(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
        elif config['optim'] == 'SGDM_C':
            optimizer = SGD_C(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
        elif config['optim'] == 'SGD_C_Only':
            optimizer = SGD_C_Only(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
        elif config['optim'] == 'SGDM_C_Only':
            optimizer = SGD_C_Only(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
        elif config['optim'] == 'Adam_C':
            optimizer = Adam_C(model.parameters(), lr = config['lr'], kappa = config['kappa'], topC = config['topC'])
        elif config['optim'] == 'Adam':
            optimizer = Adam(model.parameters(), lr = config['lr'])
        elif config['optim'] == 'SAGA':
            optimizer = SAGA(model.parameters(), n_samples = len(dataset1),lr = config['lr'])
        criterion = nn.CrossEntropyLoss().to(device)

    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
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
    ['NeuralNet'],             # model
    [100, 101, 102, 103, 104], # seeds
    ['mnist'],          # dataset
    ['SGD_C'], # optimizer
    [0.1, 0.01, 0.001, 0.0001],  # lr
    [0.9, 0.95, 0.99],  # decay
    [1, 2, 5, 10, 20],  # topC
    ['mean', 'mid', 'sum'],         # sum
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
    config['gradsum'] = ch
    config['topC'] = t
    config['kappa'] = k

    accuracy = HyperEvaluate(config)

    print("""We achieve accuracy {:7.2f}% with
        learning_rate: {:.4}
        seed: {}
        Optimizer: {}
      """.format(accuracy, l, s, o))


