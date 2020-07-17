'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
#sys.path.append('./cifar')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Model.cifar as models

from utilscifar import AverageMeter, accuracy, mkdir_p
from optimizers.optim import SGD_C, SGD, Adam_C, Adam
import itertools

from filelock import FileLock
import ray

ray.init(num_gpus=2)


#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(models.__dict__[name]))

# Validate dataset
#assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA

@ray.remote(num_gpus=2)
def HyperEvaluate(config):

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    # Datasets
    parser.add_argument('--node-ip-address=')#,192.168.2.19
    parser.add_argument('--node-manager-port=')
    parser.add_argument('--object-store-name=')
    parser.add_argument('--raylet-name=')#/tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
    parser.add_argument('--redis-address=')#192.168.2.19:6379
    parser.add_argument('--config-list=',action='store_true')#
    parser.add_argument('--temp-dir=')#/tmp/ray
    parser.add_argument('--redis-password=')#5241590000000000
    parser.add_argument('-d', '--dataset', default=config['dataset'], type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=config['lr'], type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer',default=config['optim'],type=str)
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('--model', '-a', metavar='ARCH', default=config['model'])
    parser.add_argument('--depth', type=int, default=110, help='Model depth.')
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    # Miscs
    parser.add_argument('--seed', type=int, help='manual seed', default = config['seed'])
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    #Device options
    #parser.add_argument('--gpu-id', default='0', type=str,
    #                    help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    if '_C' in args.optimizer:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr) + '_topC_' + str(args.topC) + '_decay_'+ str(args.decay) +'_'+args.gradsum
    else:
        run_id = "seed_" + str(args.seed) + '_LR_' + str(args.lr)

    MODEL_SAVE_PATH = os.path.join('Results', args.dataset, args.model + '_' + args.optimizer,'Model',run_id)
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    SAMPLES_PATH = os.path.join('Results', args.dataset, args.model + '_' + args.optimizer,'Samples',run_id)
    if not os.path.exists(SAMPLES_PATH):
        os.makedirs(SAMPLES_PATH)
    LOG_FILE_NAME = 'logs.txt'

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = "gpu"
    else:
        device = "cpu"
    best_acc = 0
    # Random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    #print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./Dataset', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./Dataset', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    #print("==> creating model '{}'".format(args.model))
    if args.model.startswith('resnext'):
        model = models.__dict__[args.model](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.model.startswith('densenet'):
        model = models.__dict__[args.model](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.model.startswith('wrn'):
        model = models.__dict__[args.model](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.model.endswith('resnet'):
        model = models.__dict__[args.model](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.model](num_classes=num_classes)

    model = torch.nn.DataParallel(model).to(device)
    #cudnn.benchmark = True
    #print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
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
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    #title = 'cifar-10-' + args.model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    #else:
        #logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        #logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    # if args.evaluate:
    #     print('\nEvaluation only')
    #     test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    #     print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    #     return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.schedule)

        #print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, offline_stats = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        val_loss, val_acc = test(testloader, model, criterion, epoch, use_cuda)
        off = offline_stats['no']*100/(sum([v for v in offline_stats.values()]) + 1e-7)
        on = offline_stats['yes']*100/(sum([v for v in offline_stats.values()]) + 1e-7)

        # append logger file
        #logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        lock = FileLock(os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME+'.new.lock'))
        with lock:
            with open(os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME),'a') as f:
                f.write(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc:7.3f} | Val. Acc: {train_acc:7.3f} |offline updates: {off:7.3f} | online udpates: {on:7.3f} |\n')
            lock.release()
        optimizer.resetOfflineStats()

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            with open(os.path.join(MODEL_SAVE_PATH,'best_model.pt'), 'wb') as f:
                torch.save(model, f)
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'acc': test_acc,
        #         'best_acc': best_acc,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, checkpoint=args.checkpoint)

    #logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

    return best_acc

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    S = {'yes':0,'no':0}
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats = optimizer.getOfflineStats()
        if stats:
            for k,v in stats.items():
                S[k]+=v

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_=batch_idx + 1,
        size_=len(trainloader)
        data_=data_time.avg
        bt_=batch_time.avg
        loss_=losses.avg
        top1_=top1.avg
        top5_=top5.avg
        # plot progress
        #print(f'{batch_}/{size_}) Data: {data_:.3f}s | Batch: {bt_:.3f}s | Loss: {loss_:.4f} | top1: {top1_: .4f} | top5: {top5_: .4f}')
    return (losses.avg, top1.avg, S)

def test(testloader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_=batch_idx + 1,
        size_=len(testloader)
        data_=data_time.avg
        bt_=batch_time.avg
        loss_=losses.avg
        top1_=top1.avg
        top5_=top5.avg
        # plot progress
        #print(f'({batch_}/{size_}) Data: {data_:.3f}s | Batch: {bt_:.3f}s | Loss: {loss_:.4f} | top1: {top1_: .4f} | top5: {top5_: .4f}')
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, schedule):
    global state
    if epoch in schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


t_models = ['resnet']
t_seeds = [100,101,102,103,104]
t_dataset = ['cifar100']
t_optim = ['SGD']#,'SGDM','Adam']
t_lr = [1e-1,1e-2,1e-3]

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
    print("""We achieve accuracy {:.3}% with
        learning_rate: {:.2}
        seed: {}
        Optimizer: {}
      """.format(100 * accuracy, hyperparameters["lr"],
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
