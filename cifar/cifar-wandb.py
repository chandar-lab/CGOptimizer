from Model.Convnets import Cifar10CnnModel, Cifar100CnnModel, Cifar100CnnModel_noDropOut

import torch.optim as optim
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C
from optimizers.optimExperimental import SAGA

import sys
import torch
import torch.nn as nn
import argparse
import os
import wandb

from itertools import product
from data_loader import load_data_subset

import Model.cifar as models

# commandline arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type = str, default = '../Dataset')
parser.add_argument('--results_path', type=str, default = '.')

parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

data_path = args.data_path
results_path = args.results_path

os.environ["WANDB_DIR"] = results_path

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
    for i, batch in enumerate(iterator):
        stats = None

       # if isinstance(model, Cifar100CnnModel) or isinstance(model, Cifar10CnnModel) or isinstance(model, resnet):
        src = batch[0]
       # else:
       #     src = batch[0].view(-1,784)
        trg = batch[1]
        src, trg = src.to(device), trg.to(device)
        output = model(src)
        optimizer.zero_grad()
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
            src = batch[0]
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

    MODEL_SAVE_PATH = os.path.join('../Results', config['dataset'], config['model'] + '_' + config['optim'], 'Model', run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if config['dataset'] == 'cifar10':
        train_iterator, valid_iterator, _, test_iterator, num_classes = load_data_subset(data_aug=1, batch_size=BATCH_SIZE,
                                                                                         workers=0, dataset='cifar10', data_target_dir = data_path, labels_per_class=5000, valid_labels_per_class=500)
    elif config['dataset'] == 'cifar100':
            train_iterator, valid_iterator, _, test_iterator, num_classes = load_data_subset(data_aug=1, batch_size=BATCH_SIZE,
                                                                                             workers=0, dataset='cifar100', data_target_dir = data_path, labels_per_class=500, valid_labels_per_class=50)

    # encoder

    if config['model'] == 'convnet':
        if config['dataset'] == 'cifar10':
            model = Cifar10CnnModel()
        elif config['dataset'] == 'cifar100':
            model = Cifar100CnnModel()

        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'])
    elif config['model'] == 'convnet_noDropOut':
        model =Cifar100CnnModel_noDropOut()
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


    best_validation_perf = float('-inf')
    best_test_perf = float('-inf')
    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss,offline_stats = train(model, train_iterator, optimizer, criterion )
        valid_loss, valid_perf = evaluate(model, valid_iterator, criterion)
        test_loss, test_perf =  evaluate(model, test_iterator, criterion)

        off = offline_stats['no']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes']*100/(sum([v for v in offline_stats.values()]) + 1e-7)

        wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss, "Val. Accuracy": valid_perf, "Test Loss": test_loss, "Test Accuracy": test_perf, "offline updates" : off, "online udpates": on})

        if config['stats']:
            gc_v_gt = optimizer.getAnalysis()
            wandb.log({'gt':gc_v_gt['gt']/gc_v_gt['count'],'gc':gc_v_gt['gc']/gc_v_gt['count']})

        optimizer.resetOfflineStats()
        #torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,config['model']+'_'+str(epoch)+'.pt'))
        if valid_perf > best_validation_perf:
           best_validation_perf = valid_perf
           torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,'best_model.pt'))

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if test_perf > best_test_perf:
            best_test_perf = test_perf

    return best_validation_perf, best_test_loss, best_test_perf


PARAM_GRID = list(product(
    ['convnet'],             # model
    [100, 101, 102, 103, 104], # seeds
    ['cifar10'],          # dataset
    ['SGD_C'], # optimizer
    [0.001],  # lr
    [0.9],  # decay
    [5, 10, 20, 50, 100],  # topC
    ['mean'],         # aggr
    [1.0],               # kappa
    [True]  #stats
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
    config['stats'] = ts
    if "_C" in o:
        config['decay'] = dec
        config['aggr'] = ch
        config['topC'] = t
        config['kappa'] = k
    else:
        config['decay'] = 0
        config['aggr'] = 'none'
        config['topC'] = 0
        config['kappa'] = 0


    val_loss, test_loss, test_ppl = HyperEvaluate(config)
    wandb.log({'Best Validation Loss': val_loss, 'Best Test Loss': test_loss, 'Best Test Perplexity': test_ppl})

