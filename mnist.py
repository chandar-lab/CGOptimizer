from torchtext import data

from Model.Convnets import ConvNetEncoder, ClassDecoder, LogisticRegression, FCLayer

import torch.optim as optim
from Utils.Eval_metric import getBLEU
from optimizers.optim import SGD_C, SGD, Adam_C, Adam

from EncoderDecoder import EncoderDecoder
import sys
#sys.path.append('/home/ml/pparth2/anaconda3/lib/python3.7/site-packages')
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
import random
import math
import csv
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import itertools

from filelock import FileLock
import ray

ray.init(num_gpus=2)

# commandline arguments

parser = argparse.ArgumentParser()
args = parser.parse_args()
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

@ray.remote(num_gpus=1)
def HyperEvaluate(config):
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-ip-address=')#,192.168.2.19
    parser.add_argument('--node-manager-port=')
    parser.add_argument('--object-store-name=')
    parser.add_argument('--raylet-name=')#/tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
    parser.add_argument('--redis-address=')#192.168.2.19:6379
    parser.add_argument('--config-list=',action='store_true')#
    parser.add_argument('--temp-dir=')#/tmp/ray
    parser.add_argument('--redis-password=')#5241590000000000
    parser.add_argument('--results_path', type=str, default = '.')

    parser.add_argument('--optimizer',type=str,default=config['optim'])
    parser.add_argument('--model',type=str,default=config['model'])
    parser.add_argument('--dataset',type=str,default=config['dataset'])
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--kappa', type=float, default = 1.0)
    parser.add_argument('--decay', type=float, default = config['decay'])
    parser.add_argument('--gradsum',type=str,default=config['gradsum'])
    parser.add_argument('--topC', type=int, default = config['topC'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--gamma', type=float, default=0.7)

    args = parser.parse_args()

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
            dataset1 = datasets.MNIST('./Dataset', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('./Dataset', train=False,
                           transform=transform)
        train_iterator = torch.utils.data.DataLoader(dataset1,batch_size=BATCH_SIZE,
                                               shuffle=True)
        valid_iterator = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE,
                                               shuffle=True)
        INPUT_DIM = 10

    N_EPOCHS = 25           # number of epochs


    MODEL_SAVE_PATH = os.path.join('Results', args.dataset, args.model + '_' + args.optimizer,'Model',run_id)
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
        elif args.optimizer == 'SGDM_C':
            optimizer = SGD_C(model.parameters(),lr = args.lr, momentum = 0.9, kappa = args.kappa, topC = args.topC)
        elif args.optimizer == 'SGDM':
            optimizer = SGD(model.parameters(),lr = args.lr, momentum = 0.9)
        elif args.optimizer == 'Adam_C':
            optimizer = Adam_C(model.parameters(), lr = args.lr, kappa = args.kappa, topC = args.topC)
        elif args.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr = args.lr)
        elif args.optimizer == 'SGD_C':
            optimizer = SGD_C(model.parameters(),lr = args.lr, decay=args.decay, topC = args.topC, sum = args.gradsum)
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

t_models = ['LR', 'NeuralNet']
t_seeds = [100, 101, 102, 103, 104]
t_dataset = ['mnist']
t_optim = ['SGD_C', 'SGDM', 'Adam']
t_lr = [1e-2, 1e-3, 1e-4]
t_decay = [0.9, 0.95, 0.99]
t_topC = [50, 10, 20, 50]
t_choice = ['sum', 'average']

best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}

for s,l,d,m,o,dec,t,ch in itertools.product(t_seeds,t_lr,t_dataset,t_models,t_optim,t_decay,t_topC,t_choice):
    config = {}
    config['model'] = m
    config['seed'] = s
    config['lr'] = l
    config['dataset'] =d
    config['optim'] = o
    config['decay'] = dec
    config['gradsum'] = ch
    config['topC'] = t
    accuracy_id = HyperEvaluate.remote(config)
    remaining_ids.append(accuracy_id)
    hyperparameters_mapping[accuracy_id] = config

###########################################################################
# Process each hyperparameter and corresponding accuracy in the order that
# they finish to store the hyperparameters with the best accuracy.

# Fetch and print the results of the tasks in the order that they complete.

while remaining_ids:
    # Use ray.wait to get the object ID of the first task that completes.
    done_ids, remaining_ids = ray.wait(remaining_ids)
    # There is only one return result by default.
    result_id = done_ids[0]

    hyperparameters = hyperparameters_mapping[result_id]
    accuracy = ray.get(result_id)
    print("""We achieve accuracy {:7.2f}% with
        learning_rate: {:.4}
        seed: {}
        Optimizer: {}
      """.format(accuracy, hyperparameters["lr"],
                 hyperparameters["seed"], hyperparameters["optim"]))
