#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import os
import shutil
import time

import torch

from utils import util
from evaluate import MultiWozEvaluator
from model.model import Model

import time
from io import open

import numpy as np
from torch.optim import Adam

from pathlib import Path

import wandb
import submitit

from itertools import product
from datetime import datetime
from filelock import FileLock

os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % args.model_path, 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def loadModelAndData(args,num):
    # Load dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(iter=num)

    # Load data
    if os.path.exists(args.decode_output):
        shutil.rmtree(args.decode_output)
        os.makedirs(args.decode_output)
    else:
        os.makedirs(args.decode_output)

    if os.path.exists(args.valid_output):
        shutil.rmtree(args.valid_output)
        os.makedirs(args.valid_output)
    else:
        os.makedirs(args.valid_output)

    # Load validation file list:
    with open('data/val_dials.json') as outfile:
        val_dials = json.load(outfile)

    # Load test file list:
    with open('data/test_dials.json') as outfile:
        test_dials = json.load(outfile)
    return model, val_dials, test_dials


def decode(args, num=1):
    model, val_dials, test_dials = loadModelAndData(args,num)
    evaluator_valid = MultiWozEvaluator("valid")
    evaluator_test = MultiWozEvaluator("test")

    start_time = time.time()
    for ii in range(1):
        if ii == 0:
            print(50 * '-' + 'GREEDY')
            model.beam_search = False
        else:
            print(50 * '-' + 'BEAM')
            model.beam_search = True

        # VALIDATION
        val_dials_gen = {}
        valid_loss = 0
        for name, val_file in val_dials.items():
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)

            valid_loss += 0
            val_dials_gen[name] = output_words

        print('Current VALID LOSS:', valid_loss)
        with open(args.valid_output + 'val_dials_gen.json', 'w') as outfile:
            json.dump(val_dials_gen, outfile)
        evaluator_valid.evaluateModel(val_dials_gen, val_dials, mode='valid')

        # TESTING
        test_dials_gen = {}
        test_loss = 0
        for name, test_file in test_dials.items():
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, test_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)
            test_loss += 0
            test_dials_gen[name] = output_words

        test_loss /= len(test_dials)
        print('Current TEST LOSS:', test_loss)
        with open(args.decode_output + 'test_dials_gen.json', 'w') as outfile:
            json.dump(test_dials_gen, outfile)
        blue_score, successes, matches  = evaluator_test.evaluateModel(test_dials_gen, test_dials, mode='test')
        lock = FileLock(os.path.join(args.model_dir+'_test_logs.txt'+'.new.lock'))
        with lock:
            with open(os.path.join(args.model_dir+'_test_logs.txt'),'a') as f:
                f.write(f'| Test BLEU: {blue_score:.3f} | Test Success: {successes:.3f} | Test Matches: {matches:.3f} \n')
                wandb.log({'Test BLEU': blue_score, 'Test Success': successes, 'Test Matches': matches} )
            lock.release()
    print('TIME:', time.time() - start_time)


def decodeWrapper(args):
    # Load config file
    with open(args.model_path + '.config') as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    args.model_path = args.model_dir+'translate.ckpt'
    args.original = args.model_path
    for ii in range(5, args.no_models + 1):
        print(70 * '-' + 'EVALUATING EPOCH %s' % ii)
        args.model_path = args.model_path + '-' + str(ii)
        try:
            decode(args,ii)
        except:
            print('cannot decode')

    args.model_path = args.original

def HyperEvaluate(config):
    parser = argparse.ArgumentParser(description='S2S')
    parser.add_argument('--no_cuda', type=util.str2bool, nargs='?', const=True, default=True, help='enables CUDA training')
    parser.add_argument('--no_models', type=int, default=5, help='how many models to evaluate')
    parser.add_argument('--original', type=str, default='model/model/', help='Original path.')

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--use_emb', type=str, default='False')
    parser.add_argument('--seed', type=int, default=config['seed'], metavar='S', help='random seed (default: 1)')
    parser.add_argument('--lr_rate', type=float, default=config['lr'])
    parser.add_argument('--optdecay', type=float, default=config['decay'])
    parser.add_argument('--topC', type=int, default=config['topC'])
    parser.add_argument('--optim', type=str, default=config['optim'])

    parser.add_argument('--beam_width', type=int, default=2, help='Beam width used in beamsearch')
    parser.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')

    parser.add_argument('--model_path', type=str, default='model/model/translate.ckpt', help='Path to a specific model checkpoint.')
    parser.add_argument('--model_dir', type=str, default='model/')
    parser.add_argument('--model_name', type=str, default='translate.ckpt')

    parser.add_argument('--valid_output', type=str, default='model/data/val_dials/', help='Validation Decoding output dir path')
    parser.add_argument('--decode_output', type=str, default='model/data/test_dials/', help='Decoding output dir path')

    args, _ = parser.parse_known_args()
    args.cuda = True

    run_id = args.optim+'_'+str(args.optdecay)+'_'+str(args.topC)+'_'+str(args.lr_rate)+'_exp_seed_{}'.format(args.seed)

    wandb.init(project="critical-gradients-mutliwozlstm-test", reinit = True)
    wandb.run.name = run_id

    wandb.config.update(config)

    print('Came here')
    args.model_dir = os.path.join('Results', 'multiwoz-lstm', 'lstm', run_id, 'model')
    #args.results_dir = os.path.join(params.outputdir, params.nlipath, params.encoder_type, run_id, results)


    torch.manual_seed(args.seed)

    decodeWrapper(args)

myparser = argparse.ArgumentParser()

myparser.add_argument('--first_launch',action='store_true')
myparser.add_argument('--is_slurm',action='store_false')
myargs=myparser.parse_args()

best_hyperparameters = None


PARAM_GRID = list(product(
    [100, 101, 102, 103, 104], # seeds
    ['sgd_c', 'sgdm_c', 'sgd','sgdm', 'adam', 'adam_c', 'rmsprop', 'rmsprop_c'], # optimizer  # lr
#    ['none'],         # aggr
#    [1.0]               # kappa
))

h_param_list = []

for param_ix in range(len(PARAM_GRID)):

    params = PARAM_GRID[param_ix]


    s, o = params
    config = {}
    config['seed'] = s
    if 'sgd' in o:
        config['lr'] = 0.1
    else:
        config['lr'] = 0.001
    config['optim'] = o
    if '_c' in o:
        config['decay'] = 0.7
        config['topC'] = 5
    else:
        config['decay'] = 0
        config['topC'] = 0
    if config not in h_param_list:
        h_param_list.append(config)

print(len(h_param_list))
if myargs.is_slurm:
    # run by submitit
    d = datetime.today()
    exp_dir = (
        Path("./dumps/")
        / "projects"
        / "crit-grad"
        / "multiwoz-lstm"
        / f"{d.strftime('%Y-%m-%d')}_rand_eval_multiwoz"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    num_gpus = 1
    workers_per_gpu = 10
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min=60,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="16G",#16G
        slurm_array_parallelism=50,
    )
    job = executor.map_array(HyperEvaluate,h_param_list)
    print('Jobs submitted!')

else:
    print("Don\'t provide the slurm argument")
