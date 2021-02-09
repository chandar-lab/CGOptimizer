import sys
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import wandb

from itertools import product

sys.path.append('..')
from optimizers.optim import SGD_C, SGD, Adam_C, Adam, RMSprop, RMSprop_C

# commandline arguments

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM/GRU/Transformer Language Model')

parser.add_argument('--data_path', type=str, default='../Dataset')
parser.add_argument('--results_path', type=str, default='.')

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

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, data_source, ntokens, bptt, criterion, ppl=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    if ppl:
        total_CE_loss = 0.
        CE_loss = nn.CrossEntropyLoss().to(device)
    # ntokens = len(corpus.dictionary)
    # if config['model'] != 'Transformer':
    eval_batch_size = 20
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
            if ppl:
                total_CE_loss += CE_loss(output, targets).item()
    return total_loss / (len(data_source) - 1)


def test(model, data_source, ntokens, bptt, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_CE_loss = 0.
    CE_loss = nn.CrossEntropyLoss().to(device)
    # ntokens = len(corpus.dictionary)
    # if config['model'] != 'Transformer':
    eval_batch_size = 20
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
            total_CE_loss += len(data) * CE_loss(output, targets).item()

    try:
        ppl = math.exp(total_CE_loss / (len(data_source) - 1))
    except OverflowError:
        ppl = float('inf')

    return total_loss / (len(data_source) - 1), ppl


def train(model, train_data, optimizer, ntokens, bptt, CLIP, batch_size, criterion):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    # ntokens = len(corpus.dictionary)
    S = {'yes': 0, 'no': 0}
    # if config['model'] != 'Transformer':
    hidden = model.init_hidden(batch_size)
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
            for k, v in stats.items():
                S[k] += v

        total_loss += loss.item()

    return total_loss / n_batches, S


# Loop over epochs.
def HyperEvaluate(config):
    import word_language_model.data as data
    import word_language_model.model as model

    # Set the random seed manually for reproducibility.
    torch.manual_seed(config['seed'])
    #    if torch.cuda.is_available():
    #       if not args.cuda:
    #          print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(os.path.join('../Dataset', config['dataset']))

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

    N_EPOCHS = 50  # number of epochs
    CLIP = 0.25  # gradient clip value    # directory name to save the models.
    if '_C' in config['optim']:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr']) + '_topC_' + str(
            config['topC']) + '_decay_' + str(config['decay']) + '_kappa_' + str(config['kappa']) + '_' + '_' + config[
                     'aggr']
    else:
        run_id = "seed_" + str(config['seed']) + '_LR_' + str(config['lr'])

    wandb.init(project="Critical-Gradients-LSTM-" + config['dataset'], reinit=True)
    wandb.run.name = run_id

    wandb.config.update(config)

    MODEL_SAVE_PATH = os.path.join('../Results', config['dataset'], config['model'] + '_' + config['optim'], 'Model',
                                   run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

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
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(
            device)
    else:
        model = model.RNNModel(config['model'], ntokens, args.emsize, args.nhid, config['layers'], args.dropout,
                               args.tied).to(device)

    criterion = nn.NLLLoss()
    if config['optim'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['lr'])
    elif config['optim'] == 'SGDM':
        optimizer = SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optim'] == 'SGD_C':
        optimizer = SGD_C(model.parameters(), lr=config['lr'], decay=config['decay'], topC=config['topC'],
                          aggr=config['aggr'])
    elif config['optim'] == 'SGDM_C':
        optimizer = SGD_C(model.parameters(), lr=config['lr'], momentum=0.9, decay=config['decay'], topC=config['topC'],
                          aggr=config['aggr'])
    elif config['optim'] == 'Adam_C':
        optimizer = Adam_C(model.parameters(), lr=config['lr'], decay=config['decay'], kappa=config['kappa'],
                           topC=config['topC'], aggr=config['aggr'])
    elif config['optim'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    elif config['optim'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=config['lr'])
    elif config['optim'] == 'RMSprop_C':
        optimizer = RMSprop_C(model.parameters(), lr=config['lr'], decay=config['decay'], kappa=config['kappa'],
                              topC=config['topC'], aggr=config['aggr'])

    best_val_ppl = float('-inf')
    best_test_ppl = float('-inf')
    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train_loss, offline_stats = train(model, train_data, optimizer, ntokens, args.bptt, args.clip, args.batch_size,
                                          criterion)
        off = offline_stats['no'] * 100 / (sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes'] * 100 / (sum([v for v in offline_stats.values()]) + 1e-7)
        val_loss, val_ppl = test(model, val_data, ntokens, args.bptt, criterion)
        test_loss, test_ppl = test(model, test_data, ntokens, args.bptt, criterion)
        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, "Validation Perplexity": val_ppl,
                   "Test Loss": test_loss, "Test Perplexity": test_ppl, "offline updates": off, "online udpates": on})

        if config['stats']:
            gc_v_gt = optimizer.getAnalysis()
            wandb.log({'gt': gc_v_gt['gt'] / gc_v_gt['count'], 'gc': gc_v_gt['gc'] / gc_v_gt['count']})

        optimizer.resetOfflineStats()

        if val_ppl < best_val_ppl:
            with open(os.path.join(MODEL_SAVE_PATH, '_best_model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val_ppl = val_ppl

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if test_ppl < best_test_ppl:
            best_test_ppl = test_ppl

    return best_val_ppl, best_test_loss, best_test_ppl


PARAM_GRID = list(product(
    ['LSTM'],  # model
    [100, 101, 102, 103, 104],  # seeds
    ['ptb-wikitext', 'wikitext'],  # dataset
    ['RMSprop_C', 'Adam_C'],  # optimizer
    [0.1, 0.01, 0.001, 0.0001, 0.00001],  # lr
    [0.7, 0.9],  # decay
    [10],  # topC
    ['mean'],  # sum
    [1.0],  # kappa
    [True],  # stats
    [2, 4]  # layers
))

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

    params = PARAM_GRID[param_ix]

    m, s, d, o, l, dec, t, ch, k, ts, ly = params

    config = {}
    config['model'] = m
    config['seed'] = s
    config['lr'] = l
    config['dataset'] = d
    config['optim'] = o
    config['stats'] = ts
    config['layers'] = ly
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

    val_ppl, test_loss, test_ppl = HyperEvaluate(config)
    wandb.log({'Best Validation Perplexity': val_ppl, 'Best Test Loss': test_loss, 'Best Test Perplexity': test_ppl})
