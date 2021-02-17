from __future__ import division, print_function, unicode_literals

import argparse
import json
import random
import time
from io import open

import numpy as np
import torch
from torch.optim import Adam

from utils import util
from model.model import Model
from pathlib import Path

import argparse
import wandb
import submitit

from itertools import product
from datetime import datetime
from evaluate import MultiWozEvaluator

import os
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, print_loss_total,print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor, name=None):
    # create an empty matrix with padding tokens
    input_tensor, input_lengths = util.padSequence(input_tensor)
    target_tensor, target_lengths = util.padSequence(target_tensor)
    bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

    loss, loss_acts, grad = model.train(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor,
                             bs_tensor, name)

    #print(loss, loss_acts)
    print_loss_total += loss
    print_act_total += loss_acts
    print_grad_total += grad

    wandb.log({'Training Loss':loss})

    model.global_step += 1
    model.sup_loss = torch.zeros(1)

    return print_loss_total, print_act_total, print_grad_total


def trainIters(model,train_dials, val_dials, n_epochs=10, args='args'):
    prev_min_loss, early_stop_count = 1 << 30, args.early_stop_count
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        print_loss_total = 0; print_grad_total = 0; print_act_total = 0  # Reset every print_every
        start_time = time.time()
        # watch out where do you put it
        # model.optimizer = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.parameters()), weight_decay=args.l2_norm)
        # model.optimizer_policy = Adam(lr=args.lr_rate, params=filter(lambda x: x.requires_grad, model.policy.parameters()), weight_decay=args.l2_norm)

        dials = train_dials.keys()
        random.shuffle(list(dials))
        input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = []
        for name in dials:
            val_file = train_dials[name]
            model.optimizer.zero_grad()
            model.optimizer_policy.zero_grad()

            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)

            if len(db_tensor) > args.batch_size:
                print_loss_total, print_act_total, print_grad_total = train(model, print_loss_total, print_act_total, print_grad_total, input_tensor, target_tensor, bs_tensor, db_tensor)
                input_tensor = [];target_tensor = [];bs_tensor = [];db_tensor = [];

        print_loss_avg = print_loss_total / len(train_dials)
        print_act_total_avg = print_act_total / len(train_dials)
        print_grad_avg = print_grad_total / len(train_dials)
        print('TIME:', time.time() - start_time)
        print('Time since %s (Epoch:%d %d%%) Loss: %.4f, Loss act: %.4f, Grad: %.4f' % (util.timeSince(start, epoch / n_epochs),
                                                            epoch, epoch / n_epochs * 100, print_loss_avg, print_act_total_avg, print_grad_avg))

        # VALIDATION
        valid_loss = 0
        val_dials_gen = {}
        evaluator_valid = MultiWozEvaluator("valid")
        for name, val_file in val_dials.items():
            input_tensor = []; target_tensor = []; bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor,
                                                                                         target_tensor, bs_tensor,
                                                                                         db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            proba, _, _ = model.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)
            proba = proba.view(-1, model.vocab_size) # flatten all predictions
            loss = model.gen_criterion(proba, target_tensor.view(-1))
            valid_loss += loss.item()
            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)
            val_dials_gen[name] = output_words

            wandb.log({'Evaluation Loss':loss.item()})
        blue_score, successes, matches = evaluator_valid.evaluateModel(val_dials_gen, val_dials, mode='valid')
        wandb.log({'Val. BLEU':blue_score, 'Val. Success':successes, 'Val. Matches': matches})

        valid_loss /= len(val_dials)
        print('Current Valid LOSS:', valid_loss)

        model.saveModel(epoch)


def loadDictionaries():
    # load data and dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    return input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index

def HyperEvaluate(config):
    print('Print!!')
    parser = argparse.ArgumentParser(description='S2S')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--vocab_size', type=int, default=400, metavar='V')

    parser.add_argument('--use_attn', type=util.str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--attention_type', type=str, default='bahdanau')
    parser.add_argument('--use_emb',  type=util.str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--emb_size', type=int, default=50)
    parser.add_argument('--hid_size_enc', type=int, default=150)
    parser.add_argument('--hid_size_dec', type=int, default=150)
    parser.add_argument('--hid_size_pol', type=int, default=150)
    parser.add_argument('--db_size', type=int, default=30)
    parser.add_argument('--bs_size', type=int, default=94)

    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--depth', type=int, default=1, help='depth of rnn')
    parser.add_argument('--max_len', type=int, default=50)

    parser.add_argument('--optim', type=str, default=config['optim'])
    parser.add_argument('--lr_rate', type=float, default=config['lr'])
    parser.add_argument('--optdecay', type=float, default=config['decay'])
    parser.add_argument('--topC', type=int, default=config['topC'])
    parser.add_argument('--lr_decay', type=float, default=0.0)
    parser.add_argument('--l2_norm', type=float, default=0.00001)
    parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

    parser.add_argument('--teacher_ratio', type=float, default=1.0, help='probability of using targets for learning')
    parser.add_argument('--dropout', type=float, default=0.0)


    #parser.add_argument('--no_cuda',  type=util.str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--seed', type=int, default=config['seed'], metavar='S', help='random seed (default: 1)')
    parser.add_argument('--train_output', type=str, default='data/train_dials/', help='Training output dir path')

    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--early_stop_count', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='model/model/')
    parser.add_argument('--model_name', type=str, default='translate.ckpt')
    parser.add_argument('--model_root', type=str, default = 'model/')

    parser.add_argument('--load_param', type=util.str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--epoch_load', type=int, default=0)

    parser.add_argument('--mode', type=str, default='train', help='training or testing: test, train, RL')

    print('Hello!')

    args, _ = parser.parse_known_args()
    args.cuda = True

    run_id = args.optim+'_'+str(args.optdecay)+'_'+str(args.topC)+'_'+str(args.lr_rate)+'_exp_seed_{}'.format(args.seed)

    wandb.init(project="critical-gradients-mutliwozlstm-sensitivity", reinit = True)
    wandb.run.name = run_id

    wandb.config.update(config)

    print('Came here')
    args.model_dir = os.path.join('Results', 'multiwoz-lstm', args.model, run_id, 'model')
    #args.results_dir = os.path.join(params.outputdir, params.nlipath, params.encoder_type, run_id, results)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index = loadDictionaries()
    print(len(output_lang_index2word))
    # Load training file list:
    with open('data/train_dials.json') as outfile:
        train_dials = json.load(outfile)

    # Load validation file list:
    with open('data/val_dials.json') as outfile:
        val_dials = json.load(outfile)

    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index).to(device)
    print(model)
    if args.load_param:
        model.loadModel(args.epoch_load)

    trainIters(model, train_dials, val_dials, n_epochs=args.max_epochs, args=args)

myparser = argparse.ArgumentParser()

myparser.add_argument('--first_launch',action='store_true')
myparser.add_argument('--is_slurm',action='store_false')
myargs=myparser.parse_args()

best_hyperparameters = None


PARAM_GRID = list(product(
    [100, 101, 102, 103, 104], # seeds
    ['adam_c'], # optimizer
    [0.001],  # lr
    [0.7],  # decay
    [5,10,20,50,100]  # topC
#    ['none'],         # aggr
#    [1.0]               # kappa
))

PARAM_GRID_ = list(product(
    [100, 101, 102, 103, 104], # seeds
    ['adam_c'], # optimizer
    [0.001],  # lr
    [0.05,0.1,0.2,0.5,0.99],  # decay
    [5]  # topC
#    ['none'],         # aggr
#    [1.0]               # kappa
))
h_param_list = {'topc':[],'decay':[]}

for param_ix in range(len(PARAM_GRID)):

    params = PARAM_GRID[param_ix]

    s, o, l, dec, t = params
    config = {}
    config['var_par'] = 'topc'
    config['seed'] = s
    if '_c' in o:
        config['lr'] = l
    else:
        if 'sgd' in o:
            config['lr'] = 0.1
        else:
            config['lr'] = 0.001
    config['optim'] = o
    if '_c' in o:
        config['decay'] = dec
        config['topC'] = t
    else:
        config['decay'] = 0
        config['topC'] = 0
    if config not in h_param_list['topc']:
        h_param_list['topc'].append(config)

for param_ix in range(len(PARAM_GRID_)):

    params = PARAM_GRID_[param_ix]

    s, o, l, dec, t = params
    config = {}
    config['var_par'] = 'decay'
    config['seed'] = s
    if '_c' in o:
        config['lr'] = l
    else:
        if 'sgd' in o:
            config['lr'] = 0.1
        else:
            config['lr'] = 0.001
    config['optim'] = o
    if '_c' in o:
        config['decay'] = dec
        config['topC'] = t
    else:
        config['decay'] = 0
        config['topC'] = 0
    if config not in h_param_list['decay']:
        h_param_list['decay'].append(config)

print(len(h_param_list['decay'])+len(h_param_list['topc']))
if myargs.is_slurm:
    # run by submitit
    d = datetime.today()
    exp_dir = (
        Path("./dumps/")
        / "projects"
        / "crit-grad"
        / "multiwoz-lstm-sensitivity"
        / f"{d.strftime('%Y-%m-%d')}_rand_eval_multiwoz"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    num_gpus = 1
    workers_per_gpu = 10
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min=300,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="47G",#16G
        slurm_array_parallelism=30,
    )
    for ke in ['decay','topc']:
        job = executor.map_array(HyperEvaluate,h_param_list[ke])
    print('Jobs submitted!')

else:
    print("Don\'t provide the slurm argument")
