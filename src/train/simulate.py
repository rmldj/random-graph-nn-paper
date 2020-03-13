import logging
import os
import pickle as pkl
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logging.basicConfig(filename='simulate.log', level=logging.INFO)

'''The file containing the main training logic.'''

# parameter numbers for the various sizes (resnet32, resnet56, resnet110)
params = {'S': 464154, 'M': 853018, 'L': 1727962}


def simulate(args):
    """
    The function performing the simulation (training + saving)
    :param args: the command line arguments. See src/main.py
    """
    if args.restype == "None":
        args.restype = None

    cmdline = ' '.join(sys.argv)
    if args.verbose:
        print_command_line(args, cmdline)

    # for approximate reproducibility
    set_seed(args)
    # handle output directories
    make_dirs(args)

    # CIFAR
    if args.cifar == '10':
        train_loader, val_loader = load_cifar10(args)
        num_outputs = 10
    elif args.cifar == '100':
        train_loader, val_loader = load_cifar100(args)
        num_outputs = 100
    else:
        raise ValueError("Unknown dataset {}".format(args.cifar))

    # load appropriate model and set basename for results/trained model
    C, basename, model, net, size = load_model(args, num_outputs)

    num_parameters = get_num_parameters(model)
    model.cuda()
    maybe_deterministic(args)

    # define loss function (criterion) and optimizer
    criterion, optimizer = set_criterion(args, model)
    # lr_scheduler
    lr_scheduler = set_lr_scheduler(args, optimizer)

    if args.verbose:
        print()
        print(args.arch, 'parameters:', num_parameters, 'size:', size, 'C:', C)
        print()

    num_epochs = args.epochs

    lr_all = np.zeros(num_epochs)
    total_time = 0
    preds = None

    for epoch in range(num_epochs):

        # train for one epoch
        epoch_time, train_losses, train_prec1 = train(train_loader, model, criterion, optimizer, epoch, args)
        total_time += epoch_time

        # update learning rate
        check_learning_rate(epoch, lr_all, lr_scheduler, optimizer, verbose=args.verbose)

        # evaluate on validation set
        if args.save_preds and epoch == num_epochs - 1:
            valid_losses, valid_prec1, preds = validate(val_loader, model, criterion, args, return_preds=True)
        else:
            valid_losses, valid_prec1 = validate(val_loader, model, criterion, args)

        # save metrics
        if epoch == 0:
            arrays = create_metric_arrays(epoch, num_epochs, train_losses, train_prec1, valid_losses, valid_prec1)
        else:
            update_all_metrics(epoch, arrays, train_losses, train_prec1, valid_losses, valid_prec1)

        if args.verbose:
            print()

    save_results(C, args, arrays, basename, cmdline, lr_all, model, net, num_epochs, num_parameters, preds, size,
                 total_time, train_losses, train_prec1, valid_losses, valid_prec1)


def get_num_parameters(model):
    """
    :param model: the pytorch model.
    :return: the total number of parameters.
    """
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in trainable])


def maybe_deterministic(args):
    """
    sets (or not) the cudnn.deterministic and cudnn.benchmark flags, depending on the command line arguments.
    :param args: command line arguments.
    :return:
    """
    if args.deterministic:
        # NOT reproducible across different GPU's
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


def print_command_line(args, cmdline):
    """
    print commandline arguments
    :param args: arguments parsed.
    :param cmdline: the raw command line.
    :return:
    """
    print(cmdline)
    print()
    print(args)
    print()


def save_results(C, args, arrays, basename, cmdline, lr_all, model, net, num_epochs, num_parameters, preds, size,
                 total_time, train_losses, train_prec1, valid_losses, valid_prec1):
    """
    saves the results.
    :param C: number of initial channels.
    :param args: command line arguments.
    :param arrays: training and validation arrays containing loss and accuracy for all epochs.
    :param basename: the basename of the model (used to form the save filename).
    :param cmdline: the commandline arguments (raw).
    :param lr_all: the learning rates.
    :param model: the pytorch model.
    :param net: the underlying graph.
    :param num_epochs: number of epochs.
    :param num_parameters: number of parameters of the model.
    :param preds: last predictions.
    :param size: The size of the model (S,M,L).
    :param total_time: the total average time used for training.
    :param train_losses: the train loss for the last epoch
    :param train_prec1: the precision for last epoch.
    :param valid_losses: the validation loss for last epoch.
    :param valid_prec1: the precision for last epoch.
    :return:
    """
    results = dict()
    train_losses_all, train_prec1_all, valid_losses_all, valid_prec1_all = arrays
    # parameters
    results['cmdline'] = cmdline
    results['args'] = args
    # architecture data
    results['arch'] = args.arch
    results['num_parameters'] = num_parameters
    results['size'] = size
    results['C'] = C
    for prop in ['meta', 'num_nodes', 'num_units', 'stages', 'edges', 'pos']:
        results[prop] = getattr(net, prop, None)
    if 'edge_weights' in dir(net):
        results['edge_weights'] = net.edge_weights()
    else:
        results['edge_weights'] = None
    # simulation results by epoch/batch
    results['lr'] = lr_all
    results['train_losses'] = train_losses_all
    results['train_accs'] = train_prec1_all
    results['test_losses'] = valid_losses_all
    results['test_accs'] = valid_prec1_all
    # final results
    results['train_loss'] = train_losses.avg
    results['train_acc'] = train_prec1.avg
    results['test_loss'] = valid_losses.avg
    results['test_acc'] = valid_prec1.avg
    results['epoch_time'] = total_time / num_epochs
    with open(os.path.join(args.results_dir, basename + '.pkl'), 'wb') as f:
        pkl.dump(results, f)
    logging.info('{} {:.3f}'.format(results['arch'], results['test_acc']))
    if args.save_model:
        torch.save({
            'state_dict': model.state_dict(),
        }, os.path.join(args.models_dir, basename + '.th'))
    if args.save_preds:
        np.save(os.path.join(args.preds_dir, basename + '.npy'), preds)


def update_all_metrics(epoch, arrays, train_losses, train_prec1, valid_losses, valid_prec1):
    '''
    updates all arrays of metrics with values computed for the current epoch.
    :param epoch: the current epoch
    :param arrays: the arrays with all training and test accuracies for all epochs.
    :param train_losses:
    :param train_prec1:
    :param valid_losses:
    :param valid_prec1:
    :return:
    '''
    train_losses_all, train_prec1_all, valid_losses_all, valid_prec1_all = arrays
    train_losses_all[epoch] = train_losses.get_array()
    train_prec1_all[epoch] = train_prec1.get_array()
    valid_losses_all[epoch] = valid_losses.get_array()
    valid_prec1_all[epoch] = valid_prec1.get_array()


def create_metric_arrays(epoch, num_epochs, train_losses, train_prec1, valid_losses, valid_prec1):
    """
    creates the arrays storing loss and accuracy for all epochs.
    :param epoch: current epoch number upon creation of the arrays (usually 0).
    :param num_epochs: total number of epochs.
    :param train_losses:
    :param train_prec1:
    :param valid_losses:
    :param valid_prec1:
    :return:
    """
    # Determine the number of batches in the training set
    arr = train_losses.get_array()
    train_nbatches = len(arr)
    train_losses_all = np.zeros((num_epochs, train_nbatches))
    train_losses_all[epoch] = arr
    train_prec1_all = np.zeros((num_epochs, train_nbatches))
    train_prec1_all[epoch] = train_prec1.get_array()
    # Determine the number of batches in the test set
    arr = valid_losses.get_array()
    valid_nbatches = len(arr)
    valid_losses_all = np.zeros((num_epochs, valid_nbatches))
    valid_losses_all[epoch] = arr
    valid_prec1_all = np.zeros((num_epochs, valid_nbatches))
    valid_prec1_all[epoch] = valid_prec1.get_array()
    return train_losses_all, train_prec1_all, valid_losses_all, valid_prec1_all


def set_lr_scheduler(args, optimizer):
    """
    Sets a learning rate scheduler.
    :param args: command line arguments.
    :param optimizer: the optimizer class, to which the scheduler is assigned.
    :return: the learning rate scheduler (torch.optim.lr_scheduler class)
    """
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[args.milestone1 - 1, args.milestone2 - 1])
    if args.arch in ['resnet1202', 'resnet110'] or args.size == 'L':
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
    return lr_scheduler


def set_criterion(args, model):
    """
    Sets the loss criterion.
    :param args: command line arguments.
    :param model: pytorch model.
    :return: the criterion and teh criterions optimizer.
    """
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return criterion, optimizer


def load_model(args, num_outputs):
    """
    Load the model corresponding to the given net architecture.
    :param args: command line argument
    :param num_outputs: output dimension (e.g. 10 for CIFAR10, 100 for CIFAR100)
    :return: initial number of channels, the net name, the pytorch model, the net architecture, the net size.
    """
    C = None
    if args.arch.startswith('resnet'):
        from src.models import resnet
        num_classes = 10
        if args.cifar == "100":
            num_classes = 100
        net = resnet.__dict__[args.arch](num_classes=num_classes)

        model = torch.nn.DataParallel(net)
        basename = args.arch
        size = 'C'
        if args.arch == 'resnet32':
            size = 'S'
        if args.arch == 'resnet56':
            size = 'M'
        if args.arch == 'resnet110':
            size = 'L'
    else:
        import importlib
        sys.path.append(args.net_dir)
        Net = getattr(importlib.import_module(args.arch), 'Net')
        if args.C is None:
            # fix C from the total number of parameters
            C = Net.get_C(params[args.size], restype=args.restype, blocktype=args.blocktype)
            net = Net(C, restype=args.restype, blocktype=args.blocktype, num_outputs=num_outputs)
            model = torch.nn.DataParallel(net)
            basename = '{}_{}'.format(args.arch, args.size)
            size = args.size
        else:
            # C set manually
            C = args.C
            net = Net(C, restype=args.restype, blocktype=args.blocktype, num_outputs=num_outputs)
            model = torch.nn.DataParallel(net)
            basename = '{}_C{}'.format(args.arch, C)
            size = 'C'
    return C, basename, model, net, size


def make_dirs(args):
    """
    Creates the directories for saving, if they do not exists already.
    """
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_model:
        os.makedirs(args.models_dir, exist_ok=True)
    if args.save_preds:
        os.makedirs(args.preds_dir, exist_ok=True)


def set_seed(args):
    """
    Sets the seed to args.seed.
    :param args: command line arguments.
    :return:
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def check_learning_rate(epoch, lr_all, lr_scheduler, optimizer, verbose):
    """
    sets and prints the current learning rate.
    :param epoch: current epoch.
    :param lr_all: arrays containing lr_rates for all epochs.
    :param lr_scheduler: the learning rate scheduler.
    :param optimizer: the optimizer.
    :param verbose: whether to be verbose.
    :return:
    """
    lr = optimizer.param_groups[0]['lr']
    if verbose:
        print('lr {:.5f}'.format(lr), end=' | test: ')
    lr_all[epoch] = lr
    lr_scheduler.step()


def load_cifar10(args):
    """
    loads the cifar10 dataset.
    :param args: command line arguments.
    :return:
    """
    CIFAR10_PATH = args.datapath
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=CIFAR10_PATH, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=CIFAR10_PATH, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def load_cifar100(args):
    """
    loads the cifar100 datasets
    :param args: command line arguments.
    :return:
    """
    CIFAR100_PATH = args.datapath
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_PATH, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_PATH, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    epoch_time = time.time() - start

    if args.verbose:
        print('Epoch: {epoch:3d}  '
              'Loss {loss.avg:.4f}  '
              'Prec@1 {top1.avg:.3f}  time {epoch_time:.2f}s'.format(
            epoch=epoch + 1, epoch_time=epoch_time, loss=losses, top1=top1), end='  ')

    return epoch_time, losses, top1


def validate(val_loader, model, criterion, args, return_preds=False):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if return_preds:
        num_val = len(val_loader.dataset)  # number of images in test set
        preds = np.zeros((num_val, 10), dtype=np.float32)
        j = 0

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            batch_size = input.size(0)
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)

            if return_preds:
                preds[j:j + batch_size] = output.data.cpu().numpy()
                j += batch_size
    if args.verbose:
        print('Loss {losses.avg:.4f}  Prec@1 {top1.avg:.3f}'.format(losses=losses, top1=top1), end='  ')

    if return_preds:
        return losses, top1, preds
    else:
        return losses, top1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(self.val)

    def save(self, filename):
        np.save(filename, np.array(self.vals))

    def get_array(self):
        return np.array(self.vals)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
