# from torchtext import data

from Model.Convnets import ConvNetEncoder, ClassDecoder, LogisticRegression, FCLayer

import torch.optim as optim
# from Utils.Eval_metric import getBLEU
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, SGD_C_Only

from EncoderDecoder import EncoderDecoder
import sys
# sys.path.append('/home/ml/pparth2/anaconda3/lib/python3.7/site-packages')
import torch
import torch.nn as nn
# import numpy as np
import argparse
import os
# import logging
# import wandb
# import random
# import math
# import csv
# import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import itertools

from filelock import FileLock
from orion.client import report_objective

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

        src = batch[0].view(-1,784)
        trg = batch[1]
        output, hidden = model(src)
        loss = criterion(output, trg)
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
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
            src = batch[0].view(-1,784)
            trg = batch[1]
            total += trg.size(0)
            output, hidden = model(src)
            loss = criterion(output, trg)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(trg.view_as(pred)).sum()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
    return epoch_loss / len(iterator), 100. * epoch_correct/ total

def HyperEvaluate():

    torch.manual_seed(args.seed)

    BATCH_SIZE = args.batch_size

    if '_C' in args.optimizer:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr) + '_topC_' + str(args.topC) + '_decay_'+ str(args.decay) +'_'+args.gradsum
    else:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr)

    if args.dataset == 'mnist':
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

    N_EPOCHS = 25           # number of epochs


    MODEL_SAVE_PATH = os.path.join(results_path, 'Results', args.dataset, args.model + '_' + args.optimizer,'Model',run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    LOG_FILE_NAME = 'logs.txt'

    # encoder
    if args.model == 'LR':
        model = LogisticRegression(784,10)
        itos_context_id = None
        itos_vocab = None
    elif args.model == 'NeuralNet':
        model = FCLayer()
        itos_context_id = None
        itos_vocab = None
    elif args.model == 'convnet':
        enc = ConvNetEncoder().to(device)
        dec = ClassDecoder().to(device)
        model = EncoderDecoder(enc,dec,data='image')
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        itos_context_id = None
        itos_vocab = None
    else:
        print('Error: Model Not There')
        sys.exit(0)

    if args.dataset == 'mnist':
        if args.optimizer == 'SGD':
            optimizer = SGD(model.parameters(),lr = args.lr)
        elif args.optimizer == 'SGDM':
            optimizer = SGD(model.parameters(),lr = args.lr, momentum = 0.9)
        elif args.optimizer == 'SGD_C':
            optimizer = SGD_C(model.parameters(),lr = args.lr, decay=args.decay, topC = args.topC, sum = args.gradsum)
        elif args.optimizer == 'SGDM_C':
            optimizer = SGD_C(model.parameters(),lr = args.lr, momentum = 0.9, decay=args.decay, topC = args.topC, sum = args.gradsum)
        elif args.optimizer == 'SGD_C_Only':
            optimizer = SGD_C_Only(model.parameters(),lr = args.lr, decay=args.decay, topC = args.topC, sum = args.gradsum)
        elif args.optimizer == 'SGDM_C_Only':
            optimizer = SGD_C_Only(model.parameters(),lr = args.lr, momentum = 0.9, decay=args.decay, topC = args.topC, sum = args.gradsum)
        elif args.optimizer == 'Adam_C':
            optimizer = Adam_C(model.parameters(), lr = args.lr, kappa = args.kappa, topC = args.topC)
        elif args.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr = args.lr)
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


        lock = FileLock(os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME+'.new.lock'))
        with lock:
            with open(os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME),'a') as f:
                f.write(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Accuracy: {valid_perf:7.3f} | offline updates: {off:7.3f} | online udpates: {on:7.3f} |\n')
        #print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Accuracy: {valid_perf:7.3f} | offline updates: {off:7.3f} | online udpates: {on:7.3f} |')
            lock.release()
        optimizer.resetOfflineStats()
        #torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,args.model+'_'+str(epoch)+'.pt'))
        if valid_perf > best_validation_perf:
           best_validation_perf = valid_perf
           torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,'best_model.pt'))

        #scheduler.step()
    return best_validation_perf

accuracy_id = HyperEvaluate()
report_objective(accuracy_id)
