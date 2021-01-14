# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
# import sys
# import logging
# import itertools
from itertools import product

import wandb

#sys.path.append(os.path.expanduser('~/Documents/CriticalGradientOptimization/optimizers'))
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, SGD_C_Only, SGD_C_single, SGD_new_momentum, SAGA, RMSprop, RMSprop_C, RMSprop_C_single, Adam_C_single
from filelock import FileLock

os.environ["WANDB_API_KEY"] = '90b23c86b7e5108683b793009567e676b1f93888'
os.environ["WANDB_MODE"] = "dryrun"
os.environ['WANDB_SILENT'] = 'true'

# commandline arguments

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM/GRU/Transformer Language Model')


parser.add_argument('--data_path', type = str, default = './Dataset')
parser.add_argument('--results_path', type=str, default = '.')


parser.add_argument('--node-ip-address=')#,192.168.2.19
parser.add_argument('--node-manager-port=')
parser.add_argument('--object-store-name=')
parser.add_argument('--raylet-name=')#/tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
parser.add_argument('--redis-address=')#192.168.2.19:6379
parser.add_argument('--config-list=',action='store_true')#
parser.add_argument('--temp-dir=')#/tmp/ray
parser.add_argument('--redis-password=')#5241590000000000
parser.add_argument('--dataset', type=str, default='ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=100,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--optimizer',type=str, default='SGD')
parser.add_argument('--kappa', type=float, default = 1.0)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--gradsum',type=str,default='sum')
parser.add_argument('--topC', type=int, default=1)

args = parser.parse_args()

data_path = args.data_path
results_path = args.results_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i,bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model,data_source,ntokens,bptt,criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    #ntokens = len(corpus.dictionary)
    #if config['model'] != 'Transformer':
    eval_batch_size =20
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(model,train_data,optimizer,ntokens,bptt,CLIP, batch_size,criterion):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    #ntokens = len(corpus.dictionary)
    S = {'yes':0,'no':0}
    #if config['model'] != 'Transformer':
    hidden = model.init_hidden(batch_size)
    start_time = time.time()
    n_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        # if config['model'] == 'Transformer':
        #     output = model(data)
        #     output = output.view(-1, ntokens)
        # else:
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()
        stats = optimizer.getOfflineStats()
        if stats:
            for k,v in stats.items():
                S[k]+=v

        total_loss += loss.item()

    return total_loss/n_batches,S


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
def HyperEvaluate(config):
    import word_language_model.data as data
    import word_language_model.model as model
    best_val_loss = None

    # Set the random seed manually for reproducibility.
    torch.manual_seed(config['seed'])
#    if torch.cuda.is_available():
 #       if not args.cuda:
  #          print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(os.path.join('Dataset',config['dataset']))

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    N_EPOCHS = 25           # number of epochs
    CLIP = 0.25               # gradient clip value    # directory name to save the models.
    if '_C' in config['optim']:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr']) + '_topC_' + str(config['topC']) + '_decay_'+ str(config['decay'])+ '_kappa_' + str(config['kappa']) +'_' +'_'+config['gradsum']
    else:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr'])

    wandb.init(project="Critical-Gradients-LSTM-" + config['dataset'], reinit = True)
    wandb.run.name = run_id

    wandb.config.update(config)

    MODEL_SAVE_PATH = os.path.join('Results', config['dataset'], config['model'] + '_' + config['optim'],'Model',run_id)

    eval_batch_size = args.batch_size
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)


    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    del corpus
    if config['model'] == 'Transformer':
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = model.RNNModel(config['model'], ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    criterion = nn.NLLLoss()
    if config['optim'] == 'SGD':
        optimizer = SGD(model.parameters(),lr = config['lr'])
    elif config['optim'] == 'SGDM':
        optimizer = SGD(model.parameters(),lr = config['lr'], momentum = 0.9)
    elif config['optim'] == 'SGDM_new':
        optimizer = SGD_new_momentum(model.parameters(),lr = config['lr'], momentum = 0.9)
    elif config['optim'] == 'SGD_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'SGD_C_single':
        optimizer = SGD_C_single(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'SGDM_C':
        optimizer = SGD_C(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'SGDM_C_single':
        optimizer = SGD_C_single(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'SGD_C_Only':
        optimizer = SGD_C_Only(model.parameters(),lr = config['lr'], decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'SGDM_C_Only':
        optimizer = SGD_C_Only(model.parameters(),lr = config['lr'], momentum = 0.9, decay=config['decay'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'Adam_C':
        optimizer = Adam_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'Adam_C_inter':
        optimizer = Adam_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'], param_level = False)
    elif config['optim'] == 'Adam_C_param':
        optimizer = Adam_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'], param_level = True)
    elif config['optim'] == 'Adam':
        optimizer = Adam(model.parameters(), lr = config['lr'])
    elif config['optim'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr = config['lr'])
    elif config['optim'] == 'RMSprop_C':
        optimizer = RMSprop_C(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'RMSprop_C_single':
        optimizer = RMSprop_C_single(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'])
    elif config['optim'] == 'Adam_C_single':
        optimizer = Adam_C_single(model.parameters(), lr = config['lr'], decay=config['decay'], kappa = config['kappa'], topC = config['topC'], sum = config['gradsum'])

        

    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        train_loss, offline_stats = train(model,train_data,optimizer,ntokens,args.bptt,args.clip,args.batch_size,criterion)
        off = offline_stats['no']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        train_time = time.time() - epoch_start_time
        val_loss = evaluate(model,val_data,ntokens,args.bptt,criterion)
        optimizer.resetOfflineStats()


        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, "offline updates" : off, "online udpates": on})


    return best_val_loss

best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}


PARAM_GRID = list(product(
    ['LSTM'],             # model
    [100, 101, 102, 103, 104], # seeds
    ['wikitext', 'ptb'],          # dataset
    ['SGDM_C_single'], # optimizer
    [0.1],  # lr
    [0.9, 0.95, 0.99],  # decay
    [1, 2, 5, 10, 20],  # topC
    ['mean', 'sum'],         # sum
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

