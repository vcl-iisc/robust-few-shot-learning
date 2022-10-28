import os
import sys
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

import wandb
from utils import LabelSmoothingLossCanonical, get_params, load_resnet_norm, seed_everything, progress_bar, AverageMeter, cosim_loss, MultiLPDataset

best_acc = 0


def get_num_classes(dataset):
    cls_ = []
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        cls_.append(y)
    
    return np.unique(cls_)


def load_data(root, batch_size=32, r_max=4, use_lp=False, return_data=False):

    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
    transforms.transforms.ColorJitter(brightness=(0.1, 0.9), contrast=(0.6, 1.4)),
    transforms.ToTensor()])

    preprocess = transforms.Compose([transforms.ToTensor()])

    if root=='CIFAR-FS':
        trainset = torchvision.datasets.ImageFolder(f'{root}/train/', transform=transform) ## Training Datasets
        trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.95), len(trainset) - int(len(trainset)*0.95)], generator=torch.Generator().manual_seed(42))        
    else:
        trainset = torchvision.datasets.ImageFolder(f'{root}/train/', transform=transform) ## Training Datasets
        valset = torchvision.datasets.ImageFolder(f'{root}/train_val/', transform=preprocess) ## Validation Data split from training data
    
    num_classes = len(get_num_classes(trainset))
    print(f'Trainset: {len(trainset)} | Valset: {len(valset)} | Classes: {num_classes}')

    if return_data:
        datasets = {'train':None, 'val':None}
        if use_lp:
            radiuses = list(range(r_max, 1, -1))

            lp_trainset = MultiLPDataset(trainset, radius=radiuses)
            datasets['train'] = lp_trainset
        else: 
            datasets['train'] = trainset
        
        datasets['val'] = valset
        return datasets, num_classes

    if use_lp:
        radiuses = list(range(r_max, 1, -1))
        lp_trainset = MultiLPDataset(trainset, radius=radiuses)
        trainloader = torch.utils.data.DataLoader(lp_trainset, batch_size=batch_size, shuffle=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    valoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)


    dataloader = {'train':trainloader, 'val':valoader}

    return dataloader, num_classes


def load_teacher_model(path, num_classes, image_size):

    state_dict = torch.load(path, map_location='cpu')
    teacher_backbone = load_resnet_norm(num_classes=num_classes, image_size=image_size)
    teacher_backbone.load_state_dict(state_dict['net'])
    print(f'Teacher Best Epoch: {state_dict["epoch"]} | {state_dict["acc"]}')
    return teacher_backbone



def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def validate_lp(loader, model, curr_rad_ind, args):
    global best_acc
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, lp_dict, targets in loader:
            lp_inputs = lp_dict[curr_rad_ind]
            inputs, lp_inputs, targets = inputs.to(device), lp_inputs.to(device, dtype=torch.float), targets.to(device)
            outputs = model(lp_inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = (correct / total) * 100.
    return acc





def select_radius(loader, model, args, th=98):
    
    curr_rad = args.radiuses[np.argmax(np.array(args.weights))] ## Current Radius with Highest Weight
    curr_rad_ind = args.radiuses.index(curr_rad)
    acc = validate_lp(loader, model, curr_rad_ind, args) ## Calculate Accuracy on This radius

    if acc > th and curr_rad != args.radiuses[-1]:
        ## Update the list
        print(f'Radius Changed | {curr_rad} achived {acc}')
        up_weights = (sorted(args.weights[-1:] + args.weights[:curr_rad_ind]) + args.weights[curr_rad_ind:-1])
        
    else:
        up_weights = args.weights

    
    args.weights = up_weights
    print(f'Curr Rad: {curr_rad} | Acc@{curr_rad}: {acc} | Weights: {args.weights}')



def train_robust(loader, model, teacher_model, optimizer, criterion, args, epoch, val_loader=None):
    
    model.train()
    teacher_model.eval()

    train_loss = 0
    correct = 0
    total = 0

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    loss_dict = {'epoch':epoch, 'ce':AverageMeter(), 'reg':AverageMeter()}

    if args.teacher_init:
        ## Fix Batch Norm layers
        print('Fixing Batch Norm Layers')
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    if args.mode == 'curr':
        select_radius(loader, model, args) ## Update the weight distribution


    for batch_idx, block in enumerate(loader):

        ##  Select A Radius
        radius = random.choices(args.radiuses, weights=args.weights)[0]

        try:
            inputs, lp_list, targets = block
            lp_inputs = lp_list[args.radiuses.index(radius)]

        except:
            inputs, targets = block

        inputs, targets = inputs.to(device), targets.to(device)

        if args.use_mixup:
            if batch_idx==0:
                print('Using Mixup')
            inputs_mix, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, device)
            
            inputs_mix, targets_a, targets_b = map(Variable, (inputs_mix,
                                                      targets_a, targets_b))

        optimizer.zero_grad()

        if args.use_mixup:
            outputs = model(inputs_mix)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        else: 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss_dict['ce'].update(loss.item(), n=inputs.size(0))

       ## Additional Losses
        with torch.no_grad():
            teacher_op = teacher_model(inputs).detach() ## On Clean Sample


        if args.reg is not None:
            lp_inputs = lp_inputs.to(device, dtype=torch.float)
            lp_outputs = model(lp_inputs) ## Get output on low-pass input

            loss_reg = cosim_loss(lp_outputs, teacher_op) ## Match Low-Pass output on student model with original teacher output
            loss_dict['reg'].update(loss_reg.item(), n=inputs.size(0))
            loss += loss_reg
        else:
            if batch_idx == 0:
                print('Not using any reg')
            loss_dict['reg'].update(0, n=inputs.size(0))


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss_dict_avg = {'epoch':epoch, 'ce':loss_dict['ce'].avg, 'reg':loss_dict['reg'].avg}

        if args.wandb:
            wandb.log(loss_dict_avg)

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return args.radiuses[np.argmax(np.array(args.weights))]



def validate(loader, model, criterion, args, epoch, save=True):
    global best_acc
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = (correct / total) * 100.
    if args.wandb and epoch>=0:
        wandb.log({'epoch':epoch, 'Test-Acc': acc})
    

    if save:
        if acc > best_acc:
            print(f'Saving.. | Acc incerased from {best_acc:.2f} to {acc:.2f}')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'weight_dist': args.weights
            }

            path = f'{args.dataset}/models/'
            if not os.path.isdir(path):
                os.mkdir(path)

            path += f'{args.run_name}_{args.dataset}_{args.architecture}_student.pth'

            print(path)
            torch.save(state, path)
            best_acc = acc

        if epoch == args.epochs:
            print(f'Saving.. | Last Epoch {best_acc:.2f} | {acc:.2f}')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'weight_dist': args.weights
            }

            path = f'{args.dataset}/models/'
            if not os.path.isdir(path):
                os.mkdir(path)

            path += f'{args.run_name}_{args.dataset}_{args.architecture}_student_last.pth'

            print(path)
            torch.save(state, path)
            best_acc = acc

    return acc


def get_curr_weights(args, lmd):
    BASE = 1.

    weights = [None]*len(args.radiuses)

    for idx in range(len(args.radiuses)):
        weights[idx] = BASE*(lmd**idx)

    return weights


def main():
    parser = argparse.ArgumentParser(description='Pretrain',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='Mini-ImageNet')
    parser.add_argument('--architecture', type=str, default='WRN2810')
    parser.add_argument('--train_and_val', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha Value for Mixup')
    parser.add_argument('--reg', default=None, help='Regularization Method to Use')
    parser.add_argument('--wandb', action='store_true', help='Log on wandb or not')
    parser.add_argument('--teacher_init', action='store_true', help='Use Teacher weights for Init of Student Networks')

    parser.add_argument('--run_name', default='baseline', type=str, help='Run Name for the Exp.')
    parser.add_argument('--num_img', default=4, type=int, help='Number of images per id in a batch')
    parser.add_argument('--num_id', default=64, type=int, help='Number of unique ids in a batch')
    parser.add_argument('--temp', default=1, type=int, help='Temperature for KD Loss')
    parser.add_argument('--r_max', default=8, type=int, help='Radius to use for Cosine LP Loss (Uniform Distribution)')
    parser.add_argument('--mode', default='uniform', type=str, help='Mode Used for Multi-Radius Learning')


    parser.add_argument('--s_epochs', default=120, type=int, help='Epochs to Train Student')

    parser.add_argument('--lmd', type=float, default=0.80, help='Lambda for curriculum weights')

    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()


    print(args)


    seed_everything(42)
    params = get_params(args.dataset, args.architecture, args.train_and_val)
    batch_size =128
    image_size = params['image_size']
    args.image_size = image_size

    if args.teacher_init:
        lr = 0.01
    else:
        lr = 0.1

    print(f'learning rate: {lr}')
    
    momentum = 0.9
    weight_decay = 1e-4 ## Check on 1e-3 and 1e-2
    cycle_length = params['cycle_length']
    epochs = args.s_epochs    
    args.epochs  = epochs

    gpu  = int(args.gpu)

    use_lp = args.reg is not None
    if use_lp:
        print(f'Radius For Training: {args.r_max} - 2')
        args.radiuses = list(range(args.r_max, 1, -1))

        if args.mode == 'curr':
            args.weights = get_curr_weights(args, lmd=args.lmd)
        else:
            args.weights = [1]*len(args.radiuses) ## For uniform sampling
        
        print(f'Radius : {args.radiuses} | Weight Dist: {args.weights}')
    
    print('='*120)
    
    dataloader, num_classes = load_data(root=args.dataset, batch_size=batch_size, use_lp=use_lp, r_max=args.r_max)
    
    print(f'Number of Classes in Model : {num_classes}')
    
    model = load_resnet_norm(num_classes, image_size).to(torch.device(f'cuda:{args.gpu}'))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingLossCanonical(smoothing=0.1)

    ## Load and Validate teacher Model

    t_path = os.path.join(args.dataset, 'models',   f'{args.dataset}_{args.architecture}_mixup.pth')

    print(f'Loading Techaer @ {t_path}')
    teacher_bacbone = load_teacher_model(path=t_path, num_classes=num_classes, image_size=image_size)
    acc_t = validate(dataloader['val'], teacher_bacbone, criterion, args, epoch=-1, save=False)
    print(f'Accuracy Teacher: {acc_t}')

    if args.teacher_init:
        model.load_state_dict(teacher_bacbone.state_dict()) ## Load Student Model with teacher weights and perform cosine LP losss
        acc_s = validate(dataloader['val'], model, criterion, args, epoch=-1, save=False)
        print(f'Accuracy Student: {acc_s} (Loaded with Teacher Weights)')
    
    for epoch in range(1, epochs+1):
        
        print(f'Epoch: {epoch} ...')
        train_robust(dataloader['train'], model, teacher_bacbone, optimizer, criterion, args, epoch, val_loader = dataloader['val'])
        validate(dataloader['val'], model, criterion, args, epoch)

        scheduler.step()




if __name__ == '__main__':
    main()