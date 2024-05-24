import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from data_loaders import cifar_dataset
from snn_models import WideResNet, VGG
from utils import *
from copy import deepcopy
from torchattacks import PGD, FGSM, AutoAttack, APGD

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers', default=0, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=32, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# data/model configuration
parser.add_argument('-data_dir', '--data_dir',default='E:\datasets',type=str,help='dataset path')
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-lamb','--lamb',default=0.000, type=float,help='regulation lamb')
parser.add_argument('-wd','--wd',default=5e-4, type=float,help='weight decay')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-tau','--tau',default=1.0, type=float,help='leaky parameter')
parser.add_argument('-en','--encode',default='',type=str,help='input encoding')

# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# if adversarial train
parser.add_argument('-atk', '--attack',default='', type=str, help='attack')
parser.add_argument('-eps','--eps',default=8, type=float, metavar='N', help='attack eps')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    T = 8 # the default timestep number is 4
    init_c=3
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = cifar_dataset(args.data_dir, use_cifar10=True)
    elif args.dataset.lower() == 'cifar100':
        train_dataset, val_dataset, znorm = cifar_dataset(args.data_dir, use_cifar10=False)
        num_labels = 100
    else:
        raise NotImplementedError

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    if args.model == "wrn16":
        model = WideResNet("wrn16", T, num_labels, znorm, args.tau)
    elif args.model == "vgg11":
        model = VGG("vgg11", T, num_labels, znorm, args.tau, init_c)
    else:
        raise NotImplementedError
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    identifier = args.model + "_snn"
    identifier += "_lamb[{:.3f}]".format(args.lamb)

    if args.attack.lower() == 'fgsm':
        atk = FGSM(model, eps=args.eps / 255)
        identifier += "_{}[{}]".format(args.attack, args.eps)
    elif args.attack.lower() == 'pgd':
        atk = PGD(model, eps=args.eps / 255, alpha=2.55 / 255, steps=7)
        identifier += "_{}[{}]".format(args.attack, args.eps)
    else:
        atk = None

    if args.encode == "poisson":
        model.poisson = True
        identifier += "_poi"

    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))

    logger.info('Start Training, lambda={}'.format(args.lamb))
    
    for epoch in range(args.epochs):
        loss1, loss2 = train_reg(model, device, train_loader, criterion, optimizer, args.lamb, atk=atk)
        logger.info('Epoch:[{}/{}]\t loss1={:.5f}\t loss2={:.5f}'.format(epoch , args.epochs, loss1, loss2))
        scheduler.step()
        acc, reg_loss = val_reg(model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\t df l2 loss={:.3f}\n'.format(epoch, args.epochs, acc, reg_loss))

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()