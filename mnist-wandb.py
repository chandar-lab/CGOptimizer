# from torchtext import data

from Model.Convnets import ConvNetEncoder, ClassDecoder, LogisticRegression, FCLayer

import torch.optim as optim
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C
from optimizers.optimExperimental import SAGA
from EncoderDecoder import EncoderDecoder
import sys
import torch
import torch.nn as nn
import argparse
import os
import wandb
from torchvision import datasets, transforms

from itertools import product
from data_loader import load_data_subset

from filelock import FileLock

os.environ["WANDB_API_KEY"] = '90b23c86b7e5108683b793009567e676b1f93888'
os.environ["WANDB_MODE"] = "dryrun"

# commandline arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type = str, default = './Dataset')
parser.add_argument('--results_path', type=str, default = '.')

parser.add_argument('--batch_size',type=int,default=64)

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
    for i, batch in enumerate(iterator):
        stats = None

        if isinstance(model, EncoderDecoder):
            src = batch[0]
        else:
            src = batch[0].view(-1,784)
        trg = batch[1]
        src, trg = src.to(device), trg.to(device)
        output, hidden = model(src)
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

    wandb.init(project="Critical-Gradients-mnist-" + str(config['model']), reinit = True)
    wandb.run.name = run_id

    wandb.config.update(config)

    MODEL_SAVE_PATH = os.path.join('Results', config['dataset'], config['model'] + '_' + config['optim'],'Model',run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if config['dataset'] == 'mnist':
        if not config['test']:
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
        else:
            train_iterator, valid_iterator, _, test_iterator, num_classes = load_data_subset(data_aug=1, batch_size=BATCH_SIZE,
                                                                                             workers=0, dataset='mnist', data_target_dir = data_path, labels_per_class=5000, valid_labels_per_class=500)


    # encoder
    if config['model'] == 'LR':
        model = LogisticRegression(784,10)
        model = model.to(device)
    elif config['model'] == 'NeuralNet':
        model = FCLayer()
        model = model.to(device)
    elif config['model'] == 'convnet':
        enc = ConvNetEncoder().to(device)
        dec = ClassDecoder().to(device)
        model = EncoderDecoder(enc,dec,data='image')
        optimizer = optim.Adadelta(model.parameters(), lr=config['lr'])
    else:
        print('Error: Model Not There')
        sys.exit(0)

    if config['dataset'] == 'mnist':
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
        criterion = nn.CrossEntropyLoss().to(device)


    best_validation_perf = float('-inf')
    best_test_perf = float('-inf')
    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss,offline_stats = train(model, train_iterator, optimizer, criterion )
        valid_loss, valid_perf = evaluate(model, valid_iterator, criterion)
        test_loss, test_perf = evaluate(model, test_iterator, criterion)

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

        #scheduler.step()
    return best_validation_perf, best_test_loss, best_test_perf



PARAM_GRID = list(product(
     ['LR'],             # model
     [100, 101, 102, 103, 104], # seeds
     ['mnist'],          # dataset
     ['Adam_C'], # optimizer
     [0.0001],  # lr
     [0.75],  # decay
     [5, 10, 20, 50, 100],  # topC
     ['mean'],         # sum
     [1.0],               # kappa
     [True]      # Stats
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

