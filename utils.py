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
# from trades import trades_loss


def train_reg(model, device, train_loader, criterion, optimizer, lamb=0.1, h=1e-2,atk=None):
    running_loss1 = 0
    running_loss2 = 0

    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            images = atk(images, labels)
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
        ##
        images.requires_grad_(True)

        outputs = model(images)
        out = outputs.gather(1, labels.unsqueeze(1)).squeeze() # choose
        batch = []
        inds = []
        for i in range(len(outputs)):
            mm, ind = torch.cat([outputs[i, :labels[i]], outputs[i, labels[i]+1:]], dim=0).max(0)
            f = torch.exp(out[i]) / (torch.exp(out[i]) + torch.exp(mm))
            batch.append(f)
            inds.append(ind.item())
        f1 = torch.stack(batch, dim=0)

        loss1 = criterion(outputs, labels)
        
        dx = grad(f1, images, grad_outputs=torch.ones_like(f1, device=device), retain_graph=True)[0]
        images.requires_grad_(False)

        v = dx.detach().sign()

        x2 = images + h*v

        outputs2 = model(x2)

        out = outputs2.gather(1, labels.unsqueeze(1)).squeeze() # choose
        batch = []
        for i in range(len(outputs2)):
            mm = torch.cat([outputs2[i, :labels[i]], outputs2[i, labels[i]+1:]], dim=0)[inds[i]]
            f = torch.exp(out[i]) / (torch.exp(out[i]) + torch.exp(mm))
            batch.append(f)
        f2 = torch.stack(batch, dim=0)

        dl = (f2-f1)/h
        
        loss2 = dl.pow(2).mean()

        loss = loss1 + lamb*loss2

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        loss.mean().backward()
        optimizer.step()
    return running_loss1, running_loss2

def val(model, test_loader, device, atk=None, dvs=False):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if dvs:
            inputs = inputs.transpose(0, 1)
            # inputs = inputs.flatten(0, 1).contiguous()
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

def val_ensemble(model, test_loader, device, atk, poi=False):
    correct = 0
    total = 0
    settings = [("bptt", 1, "zif"), ("bptt", 4., "sig"), ("bptt", 2., "atan"), ("bptr", 1., "zif")]
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        conf = []
        for setting in settings:
            model.set_attack_mode(setting)
            if atk is not None:
                atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
                if poi:
                    atk_img = 0.
                    # EOT
                    for i in range(10):
                        atk_img = atk(inputs, targets.to(device))
                else:
                    atk_img = atk(inputs, targets.to(device))
            with torch.no_grad():
                outputs = model(atk_img)
            _, predicted = outputs.cpu().max(1)
            conf.append(predicted.eq(targets).float())

        conf = torch.stack(conf, dim=0)
        conf = conf.min(0)[0]

        correct += float(conf.sum().item())
        total += float(targets.size(0))

    final_acc = 100 * correct / total
    return final_acc


def val_sparsity(model, train_loader, device, criterion):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    tt = 0.
    vdic = model.init_dic()
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        
        vdic = model._grad(vdic)
        vdic = model._rate(vdic)
        
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    for k in vdic.keys():
        vdic[k] = vdic[k]/len(train_loader)
    
    return running_loss, 100 * correct / total, vdic

def val_reg(model, test_loader, device):
    correct = 0
    total = 0
    loss = 0
    model.eval()
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        images.requires_grad_(True)

        outputs = model(images)

        _, predicted = outputs.cpu().max(1)
        outputs = torch.softmax(outputs, dim=1)
        f1 = outputs.gather(1, labels.to(device).unsqueeze(1)).squeeze() # choose
        
        dx = grad(f1, images, grad_outputs=torch.ones_like(f1, device=device))[0]
        dx = dx.view(dx.shape[0], -1)
        dx = dx.norm(dim=-1,p=2).mean()
        loss += dx.item()

        total += float(labels.size(0))
        correct += float(predicted.eq(labels.long()).sum().item())

    final_acc = 100 * correct / total
    final_loss = loss / len(test_loader)
    return final_acc, final_loss

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