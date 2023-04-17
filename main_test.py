import argparse
import copy
import json
import os
import sys
import torch
import torchattacks as attack
import data_loaders
from models import vgg
from utils import *

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=0, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=32, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')

# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv atk configuration
parser.add_argument('-atk','--attack',default='', type=str, help='attack')
parser.add_argument('-eps','--eps',default=8, type=float, metavar='N', help='attack eps')

# only pgd
parser.add_argument('-alpha', '--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps', '--steps',default=7,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-bb', '--bbmodel',default='',type=str,help='black box model')
parser.add_argument('-stdout', '--stdout',default='',type=str,help='log file')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=True)
    elif args.dataset.lower() == 'cifar100':
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=False)
        num_labels = 100

    seed_all(args.seed)

    log_dir = '%s-results'% (args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger(os.path.join(log_dir, '%s.log'%(args.identifier+args.suffix)))
    logger.info('start testing!')

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = vgg(num_labels, znorm)
    model.to(device)

    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    
    for atk_config in config:
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(atkmodel, eps=args.eps / 255)
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps)
        elif args.attack.lower() == 'bim':
            atk = attack.BIM(atkmodel, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps)
        else:
            atk = None
        
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        
        if atk is not None:
            acc = val_success_rate(model, test_loader, device, atk)
            logger.info(acc)
        else:
            acc = val(model, test_loader, device, atk)
            logger.info(acc)


if __name__ == "__main__":
    main()