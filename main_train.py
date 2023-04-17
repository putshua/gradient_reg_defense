import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import torchattacks.attack as attack
import data_loaders
from models import vgg
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=2, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-lamb','--lamb',default=0.1, type=float,help='regulation lamb')
parser.add_argument('-wd','--wd',default=5e-4, type=float,help='weight decay')
# parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')

# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        img_size = 32
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=True)
    elif args.dataset.lower() == 'cifar100':
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=False)
        num_labels = 100
        img_size = 32
    elif args.dataset.lower() == "catsdogs":
        train_dataset = data_loaders.CatsDogs(train=True)
        val_dataset = data_loaders.CatsDogs(train=False)
        num_labels = 2
        img_size = 224
        znorm = None

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = vgg(num_labels, znorm, img_size=img_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    identifier = "vgg16"
    identifier += "_lamb[{:.2f}]".format(args.lamb)
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))

    logger.info('Start Training, lambda={}'.format(args.lamb))
    
    for epoch in range(args.epochs):
        loss1, loss2 = train_reg(model, device, train_loader, criterion, optimizer, args.lamb)
        logger.info('Epoch:[{}/{}]\t loss1={:.5f}\t loss2={:.5f}'.format(epoch , args.epochs, loss1, loss2))
        scheduler.step()
        acc = val(model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch, args.epochs, acc))

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()