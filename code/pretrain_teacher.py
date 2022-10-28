# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import argparse
import numpy as np

import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

from utils import LabelSmoothingLossCanonical, get_params, load_resnet_norm, seed_everything, progress_bar

best_acc = 0

def load_data(root, batch_size=32):

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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valoader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    dataloader = {'train':trainloader, 'val':valoader}

    return dataloader, num_classes



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

def get_num_classes(dataset):
    cls_ = []
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        cls_.append(y)
    
    return np.unique(cls_)



def train(loader, model, optimizer, criterion, args):
    
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    for batch_idx, block in enumerate(loader):

        inputs, targets = block
        inputs, targets = inputs.to(device), targets.to(device)

        if args.use_mixup:
            if batch_idx==0:
                print('Using Mixup')
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, device)
            
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        optimizer.zero_grad()
        outputs = model(inputs)

        if args.use_mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



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

    if save:
        if acc > best_acc:

            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            path = f'{args.dataset}/models/'
            if not os.path.isdir(path):
                os.mkdir(path)

            if args.use_mixup:
                path += f'{args.dataset}_{args.architecture}_mixup.pth'
            else: 
                path += f'{args.dataset}_{args.architecture}.pth'

            print(f'Saving @ {path} | Acc incerased from {best_acc:.2f} to {acc:.2f}')

            torch.save(state, path)
            best_acc = acc
    return acc



def main():
    parser = argparse.ArgumentParser(description='Pretrain',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='Mini-ImageNet')
    parser.add_argument('--architecture', type=str, default='WRN2810')
    parser.add_argument('--train_and_val', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha Value for Mixup')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train')

    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    seed_everything(42)
    params = get_params(args.dataset, args.architecture, args.train_and_val)
    batch_size =128
    image_size = params['image_size']
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    cycle_length = params['cycle_length']
    num_cycles = 2
    cycle_multiplier = 2
    # epochs = int(cycle_length * (1 - (cycle_multiplier ** num_cycles)) \
    #     / (1 - cycle_multiplier))
    epochs = args.epochs

    
    dataloader, num_classes = load_data(root=args.dataset, batch_size=batch_size)
    print(f'Number of Classes in Model : {num_classes}')
    
    model = load_resnet_norm(num_classes, image_size).to(torch.device(f'cuda:{args.gpu}'))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingLossCanonical(smoothing=0.1)


    for epoch in range(1, epochs+1):
        
        print(f'Epoch: {epoch} ...')

        train(dataloader['train'], model, optimizer, criterion, args)
        validate(dataloader['val'], model, criterion, args, epoch)
        scheduler.step()
        

if __name__ == '__main__':
    main()