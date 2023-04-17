import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import os
import numpy as np
import random
import logging
from torch.autograd import grad

def train(model, device, train_loader, criterion, optimizer):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def train_reg(model, device, train_loader, criterion, optimizer, lamb=0.1, h=1e-2):
    running_loss1 = 0
    running_loss2 = 0
    l2_loss = nn.MSELoss()
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if lamb > 0:
            ##
            images.requires_grad_(True)
            outputs = model(images)
            f1 = outputs.gather(1, labels.unsqueeze(1)).squeeze() # choose
            # f1 = outputs.mean(1) # mean
            loss1 = criterion(outputs, labels)
            
            dx = grad(f1, images, grad_outputs=torch.ones_like(f1), retain_graph=True)[0]
            images.requires_grad_(False)
            
            v = dx.view(dx.shape[0], -1)

            # nv = v.norm(2, dim=-1, keepdim=True)
            # nz = nv.view(-1) > 0 # non-zero
            # v[nz] = v[nz].div(nv[nz])

            v = v/v.norm(2, dim=-1, keepdim=True)
            
            v = v.view(dx.shape).detach()
            x2 = images + h*v

            outputs = model(x2)
            f2 = outputs.gather(1, labels.unsqueeze(1)).squeeze()

            dl = (f2-f1)/h # This is the finite difference approximation of the directional derivative of the loss
            
            loss2 = dl.pow(2).mean()/2
            loss = loss1 + lamb*loss2
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss1 = loss
            loss2 = torch.tensor(0)
        ##
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        loss.mean().backward()
        optimizer.step()
    return running_loss1, running_loss2


def val(model, test_loader, device, atk=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc

# todo
def val_reg(model, test_loader, device):
    pass
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        labels = labels.to(device)
        images = images.to(device)
        
        outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)

        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc

def val_success_rate(model, test_loader, device, atk=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        mask = predicted.eq(targets).float()
        
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        
        predicted = ~(predicted.eq(targets))
        total += mask.sum()
        correct += (predicted.float()*mask).sum()

    final_acc = 100 * correct / total
    return final_acc.item()


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger