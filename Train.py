from torchtext import data
from Model.RNN import RecurrentEncoder, Encoder, AttnDecoder, Decoder
#from Model.Transformer import TransformerModel
from Model.Convnets import ConvNetEncoder, ClassDecoder, LogisticRegression, FCLayer
#from DatasetUtils.DataIterator import MultiWoZ, PersonaChat
import torch.optim as optim
from Utils.Eval_metric import getBLEU
from optimizers.optim import SGD_C, SGD, Adam_C, Adam
#from Utils.optim import GradualWarmupScheduler
#from Utils.TransformerUtils import create_masks
from EncoderDecoder import EncoderDecoder
import sys
sys.path.append('/home/ml/pparth2/anaconda3/lib/python3.7/site-packages')
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

ray.init()

# commandline arguments

parser = argparse.ArgumentParser()
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#np.random.seed(args.seed)
def train(model, iterator, optimizer, criterion, clip, itos_vocab=None, itos_context_id=None):
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
        if model.data == 'text':
            src = batch.Context.to(device)
            trg = batch.Target.to(device)

            optimizer.zero_grad()

            # trg is of shape [sequence_len, batch_size]
            # output is of shape [sequence_len, batch_size, output_dim]
            if model.type == 'transformer':
                src = src.transpose(0,1)
                trg = trg.transpose(0,1)
                trg_input = trg[:,:-1]
                src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
                output,hidden = model(src, src_mask, trg_input, trg_mask)
                ys = trg[:,1:].contiguous().view(-1)
                loss = criterion(output.view(-1, output.size(-1)), ys)
            else:
                output,hidden = model(src, trg)
                loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            # for b_id,hidden_state in zip(batch.context_id.squeeze(0),hidden.squeeze(0)):
            #     str_hidden = [itos_context_id[b_id]]
            #     str_hidden += [str(x.item()) for x in hidden_state]
            #     hidden_saver_train.write(','.join(str_hidden)+'\n')


            # backward pass
        elif model.data == 'image':
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

def evaluate(model, iterator, criterion, itos_vocab = None, itos_context_id = None, sample_saver = None):
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
            if model.data == 'text':
                epoch_correct = 'NA'
                src = batch.Context.to(device)
                trg = batch.Target.to(device)
                if model.type == 'transformer':
                    src = src.transpose(0,1)
                    trg = trg.transpose(0,1)
                    trg_input = trg[:,:-1]
                    src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
                    output,hidden = model(src, src_mask, trg_input, trg_mask)
                    ys = trg[:,1:].contiguous().view(-1)
                    loss = criterion(output.view(-1, output.size(-1)), ys)
                    output = output.permute(1,0,2)
                else:
                    output , hidden = model(src,trg,0)  # turn off the teacher forcing
                    loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
                top1 = output.max(2)[1].squeeze(0)
                # for b_id,hidden_state in zip(batch.context_id.squeeze(0),hidden.squeeze(0)):
                #     str_hidden = [itos_context_id[b_id]]
                #     str_hidden += [str(x.item()) for x in hidden_state]
                #     hidden_saver_eval.write(','.join(str_hidden)+'\n')
                for b_index in range(len(batch)):
                    c = ' '.join([itos_vocab[idx.item()] for idx in batch.Context[:,b_index]])
                    t = ' '.join([itos_vocab[idx.item()] for idx in batch.Target[:,b_index]])
                    model_res = ' '.join([itos_vocab[idx.item()] for idx in top1[:,b_index]])
                    sample_saver.write('Context: '+c +'\n'+'Model_Response: '+ model_res +'\n' +'Target: ' + t +'\n\n')
                # loss function works only 2d logits, 1d targets
                # so flatten the trg, output tensors. Ignore the <sos> token
                # trg shape shape should be [(sequence_len - 1) * batch_size]
                # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            elif model.data == 'image':
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

@ray.remote
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
    parser.add_argument('--s2s_hidden_size', type=int, default = 256)
    parser.add_argument('--s2s_embedding_size',type=int, default = 128)
    parser.add_argument('--transformer_dropout',type=float, default = 0.2)
    parser.add_argument('--transformer_hidden_dim',type=int, default = 512)
    parser.add_argument('--transformer_embedding_dim',type=int, default = 512)
    parser.add_argument('--transformer_n_layers',type=int, default = 2)
    parser.add_argument('--transformer_n_head',type=int, default = 2)
    parser.add_argument('--optimizer',type=str,default=config['optim'])
    parser.add_argument('--model',type=str,default=config['model'])
    parser.add_argument('--dataset',type=str,default=config['dataset'])
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--kappa', type=float, default = 1.0)
    parser.add_argument('--topC', type=int, default = 10)
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--gamma', type=float, default=0.7)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    MAX_LENGTH = 101
    BATCH_SIZE = args.batch_size
    if '_C' in args.optimizer:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr) + '_topC_' + str(args.topC) + '_kappa_'+ str(args.kappa)
    else:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr)
    if args.dataset == 'MultiWoZ':
        train_iterator, valid_iterator, test_iterator, pad_idx, INPUT_DIM, itos_vocab, itos_context_id = MultiWoZ(batch_size = BATCH_SIZE ,max_length = MAX_LENGTH)
    elif args.dataset == 'PersonaChat':
        train_iterator, valid_iterator, pad_idx, INPUT_DIM, itos_vocab, itos_context_id = PersonaChat(batch_size = BATCH_SIZE, max_length = MAX_LENGTH)
    elif args.dataset == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        with FileLock(os.path.expanduser("~/data.lock")):
            dataset1 = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                           transform=transform)
        train_iterator = torch.utils.data.DataLoader(dataset1,batch_size=BATCH_SIZE,
                                               shuffle=True)
        valid_iterator = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE,
                                               shuffle=True)
        INPUT_DIM = 10

    N_EPOCHS = 25           # number of epochs
    CLIP = 10               # gradient clip value    # directory name to save the models.

    MODEL_SAVE_PATH = os.path.join('Results', args.dataset, args.model + '_' + args.optimizer,'Model',run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    SAMPLES_PATH = os.path.join('Results', args.dataset, args.model + '_' + args.optimizer,'Samples',run_id)
    if not os.path.exists(SAMPLES_PATH):
        os.makedirs(SAMPLES_PATH)
    LOG_FILE_NAME = 'logs.txt'
    logging.basicConfig(filename=os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    OUTPUT_DIM = INPUT_DIM
    ENC_EMB_DIM = args.s2s_embedding_size # encoder embedding size
    DEC_EMB_DIM = args.s2s_embedding_size   # decoder embedding size (can be different from encoder embedding size)
    HID_DIM = args.s2s_hidden_size       # hidden dimension (must be same for encoder & decoder)
    N_LAYERS = 2        # number of rnn layers (must be same for encoder & decoder)
    HRED_N_LAYERS = 2
    ENC_DROPOUT = 0   # encoder dropout
    DEC_DROPOUT = 0   # decoder dropout (can be different from encoder droput)

    #TransformerParameters

    emsize = args.transformer_embedding_dim # 200 embedding dimension
    nhid = args.transformer_hidden_dim #200 the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.transformer_n_layers #2 the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = args.transformer_n_head #2 the number of heads in the multiheadattention models
    dropout = args.transformer_dropout # 0.2 the dropout value


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
    elif args.model == 'seq2seq':
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
        dec = Decoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
        model = EncoderDecoder(enc, dec, attn = False).to(device)
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.model == 'hred':
        enc = RecurrentEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
        dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
        model = EncoderDecoder(enc, dec, attn = True).to(device)
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.model == 'seq2seq_attn':
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HRED_N_LAYERS, ENC_DROPOUT).to(device)
        dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, HRED_N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
        model = EncoderDecoder(enc, dec, attn = True).to(device)
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.model == 'bilstm_attn':
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bi_directional = True).to(device)
        dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
        model = EncoderDecoder(enc, dec, attn = True).to(device)
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.model == 'transformer':
        model = TransformerModel(INPUT_DIM, emsize, nhead, nlayers, dropout).to(device).to(device)
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=2)
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
            optimizer = SGD_C(model.parameters(),lr = args.lr, kappa = args.kappa, topC = args.topC)
        criterion = nn.CrossEntropyLoss().to(device)

    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    # loss function calculates the average loss per token
    # passing the <pad> token to ignore_idx argument, we will ignore loss whenever the target token is <pad>


    best_validation_perf = float('-inf')

    for epoch in range(N_EPOCHS):
        sample_saver_eval = None
        valid_perf = 'NA'
        if model.data == 'text':
            sample_saver_eval = open(os.path.join(SAMPLES_PATH,"samples_valid_" +str(epoch) +'.txt'),'w')
        train_loss,offline_stats = train(model, train_iterator, optimizer, criterion, CLIP, itos_vocab = itos_vocab, itos_context_id = itos_context_id )
        valid_loss, valid_perf = evaluate(model, valid_iterator, criterion, itos_vocab = itos_vocab, itos_context_id = itos_context_id, sample_saver = sample_saver_eval)
        off = offline_stats['no']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        if sample_saver_eval != None:
            sample_saver_eval.close()
            valid_perf = getBLEU(sample_saver_eval.name)
        if model.data == 'text':
            logging.info(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_perf:7.3f} |')
            #print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_perf:7.3f} |')
        else:
            logging.info(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Accuracy: {valid_perf:7.3f} | offline updates: {off:7.3f} | online udpates: {on:7.3f} |')
            #print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. Accuracy: {valid_perf:7.3f} | offline updates: {off:7.3f} | online udpates: {on:7.3f} |')
        optimizer.resetOfflineStats()
        #torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,args.model+'_'+str(epoch)+'.pt'))
        if valid_perf > best_validation_perf:
           best_validation_perf = valid_perf
           torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,args.model+'_best_model.pt'))

        #scheduler.step()
    return best_validation_perf

t_models = ['NeuralNet']
t_seeds = [100,101,102,103,104]
t_dataset = ['mnist']
t_optim = ['SGD','SGDM','Adam']
t_lr = [1e-2,1e-3,1e-4]

best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}

for s,l,d,m,o in itertools.product(t_seeds,t_lr,t_dataset,t_models,t_optim):
    config = {}
    config['model'] = m
    config['seed'] = s
    config['lr'] = l
    config['dataset'] =d
    config['optim'] = o
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

# Record the best performing set of hyperparameters.
# print("""Best accuracy over {} trials was {:.3} with
#       learning_rate: {:.2}
#       batch_size: {}
#       momentum: {:.2}
#       """.format(num_evaluations, 100 * best_accuracy,
#                  best_hyperparameters["learning_rate"],
#                  best_hyperparameters["batch_size"],
#                  best_hyperparameters["momentum"]))
