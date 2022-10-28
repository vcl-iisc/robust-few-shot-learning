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


import argparse
import os
import random
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.callbacks import *
from fastai.distributed import *
from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import WideResNet as wrn
from frequencyHelper import \
    generateDataWithDifferentFrequencies_3Channel as freq_3t
from scipy import stats
# resnet152 with adaptive pool at the end of the feature extractor
from torchvision.models import resnet152
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def seed_everything(seed: int):
     
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_params(dataset, architecture, train_and_val):

    params = {}

    if dataset in ['Mini-ImageNet', 'Tiered-ImageNet']:
        params['image_size'] = 84
    elif dataset in ['CIFAR-FS', 'FC-100']:
        params['image_size'] = 32

    if train_and_val:
        params['csv_file_name'] = 'train_and_val.csv'
        params['parameter_file_name'] = dataset + '_' + architecture + '_TV'
    else:
        params['csv_file_name'] = 'train.csv'
        params['parameter_file_name'] = dataset + '_' + architecture
    params['parameter_path'] = os.path.join(dataset, 'models',
        params['parameter_file_name'] + '.pth')

    params['cycle_length'] = 40
    params['relu'] = True
    params['lambda'] = 1.

    return params

def get_transformations(image_size):
    
    transformations = [
        flip_lr(p=0.5),
        *rand_pad(padding=4, size=image_size, mode='reflection'),
        brightness(change=(0.1, 0.9)),
        contrast(scale=(0.6, 1.4))
        ]
    return transformations

def get_data(dataset, train_file_name, validation_file_name, image_size,
    batch_size):

    train_list = ImageList.from_csv(dataset, train_file_name)
    validation_list = ImageList.from_csv(dataset, validation_file_name)
    loaders = ItemLists(dataset, train_list, validation_list) \
        .label_from_df() \
        .transform((get_transformations(image_size), []), size=image_size) \
        .databunch(bs=batch_size, num_workers=4) \
        .normalize(imagenet_stats)
    return loaders

class FewShotDataset(torch.utils.data.Dataset):

    def save_images(self, dataset, file_name):

        data = pd.read_csv(os.path.join(dataset, file_name))
        classes = np.unique(data['c']).tolist()
        self.images = {
            cls : data.loc[data['c'] == cls]['fn'].tolist() for cls in classes
            }

    def __init__(self, dataset, file_name, image_size, way, support_shot,
        query_shot):

        self.dataset = dataset
        self.save_images(dataset, file_name)
        self.image_size = image_size
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot

    def get_way(self):

        return self.way

    def get_query_shot(self, classes):

        query_shot = {cls : self.query_shot for cls in classes}
        return query_shot

    def get_support_shot(self, classes, query_shot):

        support_shot = {cls : self.support_shot for cls in classes}
        return support_shot

    def __getitem__(self, idx):

        found_episode = False
        while not found_episode:
            found_episode = True
            way = self.get_way()
            classes = np.random.choice(list(self.images.keys()), way,
                replace=False)
            classes = sorted(classes)
            query_shot = self.get_query_shot(classes)
            support_shot = self.get_support_shot(classes, query_shot)
            support = dict(images=[], classes=[])
            query = dict(images=[], classes=[])
            for cls in classes:
                try:
                    images = np.random.choice(self.images[cls],
                        support_shot[cls] + query_shot[cls], replace=False)
                except:
                    found_episode = False
                    break
                support['images'] += images[: support_shot[cls]].tolist()
                support['classes'] += ([cls] * support_shot[cls])
                query['images'] += images[support_shot[cls] :].tolist()
                query['classes'] += ([cls] * query_shot[cls])

        support = pd.DataFrame(
            {'fn' : support['images'], 'c' : support['classes']}
            )
        query = pd.DataFrame(
            {'fn' : query['images'], 'c' : query['classes']}
            )

        support = ImageList.from_df(support, self.dataset).split_none() \
            .label_from_df().train
        query = ImageList.from_df(query, self.dataset).split_none() \
            .label_from_df().train
        for ind in range(len(query.y.items)):
            query.y.items[ind] = \
                support.y.classes.index(query.y.classes[query.y.items[ind]])
        query.y.classes = support.y.classes

        support = ItemLists(self.dataset, support, support) \
            .transform((get_transformations(self.image_size), []),
                size=self.image_size) \
            .databunch(bs=len(support), num_workers=0) 
            # .normalize(imagenet_stats) ## Will get normalized twice for torch pre-trained model
        query = ItemLists(self.dataset, query, query) \
            .transform((get_transformations(self.image_size), []),
                size=self.image_size) \
            .databunch(bs=len(query), num_workers=0)
            # .normalize(imagenet_stats)

        return support, query

def micro_forward(model, x, y, loss_func=None, loss_coef=None):

    num = x.size(0)
    yhs = []
    fs = []
    model.zero_grad()
    for x, y in zip(torch.split(x, 75), torch.split(y, 75)):
        yh = model(x)
        yhs.append(yh)
        if loss_func:
            f = x.size(0) * loss_func(yh, y) / num
            (loss_coef * f).backward()
            fs.append(f)
    yh = torch.cat(yhs)
    if loss_func:
        f = torch.stack(fs).sum()
        return yh, f
    else:
        return yh


def cosim_loss(x,y):
       return 1 -(F.cosine_similarity(x,y)).mean()


def micro_forward_reg(model, x, args):
    
    num = x.shape[0]
    yhs = []
    fs = []

    model.zero_grad()
    for x in torch.split(x, 75):
        yh = model(x)
        yhs.append(yh)

        ## Get Low pass version of x
        radius = random.choices(args.ensemble_rad, weights=args.weights)[0]

        x_lp, _ = get_freq(x, r=radius)
        x_lp = x_lp.float().cuda()
        yh_lp = model(x_lp)


        if args.reg.split('_')[0] == 'cosine':
            # print(radius, args.weights[args.radiuses.index(radius)])

            f = x.size(0) * cosim_loss(yh_lp, yh.detach()) / num
            (2 * f).backward()
            fs.append(f)

    yh = torch.cat(yhs)
    f = torch.stack(fs).sum()
    return yh, f

def get_classifier(yh, y):

    classifier = torch.zeros((y.unique().size(0), yh.size(1))).cuda()
    for cls in torch.sort(y.unique())[0]:
        classifier[cls] = yh[y == cls].mean(dim=0)
    classifier = torch.nn.functional.normalize(classifier)
    return classifier



class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FewShotModel(torch.nn.Module):

    def __init__(self, backbone, support, relu):

        super().__init__()

        '''
        Support-Based Initialization
        Given the pre-trained model (backbone), we append a ReLU layer, an
        l2-normalization layer and a fully-connected layer that takes the
        logits of the backbone as input and predicts the few-shot labels. We
        calculate the per-class l2-normalized average features to initialize
        the weights of the fully-connected layer, with the biases set to 0.
        '''

        self.backbone = deepcopy(backbone)
        self.backbone.eval()

        self.relu = relu

        with torch.no_grad():
            for x, y in support.valid_dl:
                break
            yh = micro_forward(self.backbone, x, y)
            if self.relu:
                yh = torch.relu(yh)
            classifier = get_classifier(yh, y)

            self.classifier = torch.nn.Linear(classifier.size(1),
                classifier.size(0))
            self.classifier = self.classifier.cuda()
            self.classifier.weight.data.copy_(classifier)
            self.classifier.bias.zero_()

    def forward(self, x):

        x = self.backbone(x)
        if self.relu:
            x = torch.relu(x)
        x = torch.nn.functional.normalize(x)
        x = self.classifier(x)
        return x

def validate(model, data):

    model.eval()
    num = 0
    correct = 0
    with torch.no_grad():
        for x, y in data.valid_dl:
            yh = micro_forward(model, x, y)
            num += x.size(0)
            correct += (yh.argmax(dim=1) == y).sum().item()
    accuracy = 100. * correct / num
    return accuracy

def validate_lp(model, data, radius):
    
    model.eval()
    num = 0
    correct = 0
    with torch.no_grad():
        for x, y in data.valid_dl:
            
            ## Get the LP sample
            x, _ = get_freq(x, r=radius)
            x = x.float().cuda()

            yh = micro_forward(model, x, y)
            num += x.size(0)
            correct += (yh.argmax(dim=1) == y).sum().item()
    accuracy = 100. * correct / num
    return accuracy


def get_freq(data, r):

    images = data.detach().cpu()

    images = images.permute(0,2,3,1)
    img_l, img_h = freq_3t(images, r=r)
    img_l, img_h = torch.from_numpy(np.transpose(img_l, (0,3,1,2))), torch.from_numpy(np.transpose(img_h, (0,3,1,2)))
    return img_l, img_h

def get_adv(model, data, attack, args, radius=None):
    model.eval()
    num = 0
    correct = 0 

    for x, y in data.valid_dl:
        x, y = x.cuda(), y.cuda()
        x_adv = attack(x, y)
        model.eval()

        if radius is not None:
            x_adv, _ = get_freq(x_adv, r=radius)
            x_adv = x_adv.float().cuda()

        yh = micro_forward(model, x_adv, y)
        num += x_adv.shape[0]
        correct += (yh.argmax(dim=1) == y).sum().item()
    accuracy = 100. * correct / num

    return accuracy


def validate_adv(model, data, attack, args, r=None):
    if r is None:
        acc = get_adv(model, data, attack, args)
        return acc
    else:
        # print(f'Radius Used for calcualting ADV LP: {r}')  
        adv_acc = get_adv(model, data, attack, args, radius=r)
        return adv_acc



def validate_ensemble(model, data, attack, radiuses, args=None):
    model.eval()

    num = 0
    correct = 0
    num_adv = 0
    correct_adv = 0
    for x, y in data.valid_dl:
        for x, y in zip(torch.split(x, 5), torch.split(y, 5)):
            clean_logits = []
            adv_logits = []    
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y)
            # model.eval()
            
            for r in radiuses:
                idx = args.ensemble_rad.index(r)
                w = args.weights[idx] ## Weights corresponding to radius 'r
                # print(f'Ensembling radius: {r} | Weight: {w}')

                ## Pass Clean Sample Through a cerain radius and evaluate
                x_lp, _ = get_freq(x, r=r)
                x_lp = x_lp.float().cuda()
                yh = micro_forward(model, x_lp, y)
                clean_logits.append(w * yh) ## Weighting with distribution weights
                

                ## Pass Adversarial Sample Through a cerain radius and evaluate
                x_lp_adv, _ = get_freq(x_adv, r=r)
                x_lp_adv = x_lp_adv.float().cuda()
                yh_adv = micro_forward(model, x_lp_adv, y)
                adv_logits.append(w * yh_adv) ## Weighting with distribution weights
        
            avg_clean_logits = sum(clean_logits)/len(radiuses)    
            avg_adv_logits = sum(adv_logits)/len(radiuses) 
            
            num += x_lp_adv.shape[0]
            correct += (avg_clean_logits.argmax(dim=1) == y).sum().item()
            num_adv += x_lp_adv.shape[0]
            correct_adv += (avg_adv_logits.argmax(dim=1) == y).sum().item()       

    # print(num, correct, num_adv, correct_adv)
    # exit(0) 
    
    accuracy = 100. * correct / num
    adv_accuracy = 100. * correct_adv / num_adv    

    return accuracy, adv_accuracy


cross_entropy = torch.nn.functional.cross_entropy

def entropy(yh, y):

    p = torch.nn.functional.softmax(yh, dim=1)
    log_p = torch.nn.functional.log_softmax(yh, dim=1)
    loss = - (p * log_p).sum(dim=1).mean()
    return loss

class Flatten(torch.nn.Module):

    def forward(self, x):

        x = x.view(x.size(0), -1)
        return x

class Conv64(torch.nn.Module):

    @staticmethod
    def conv_bn(in_channels, out_channels, kernel_size, padding, pool):

        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(pool)
            )
        return model

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = torch.nn.Sequential(
            self.conv_bn(3, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            self.conv_bn(64, 64, 3, 1, 2),
            Flatten(),
            torch.nn.Linear(64 * (int(image_size / 16) ** 2), num_classes)
            )

    def forward(self, x):

        x = self.model(x)
        return x

class ResNet12(torch.nn.Module):

    class Block(nn.Module):

        def __init__(self, in_channels, out_channels):

            super().__init__()

            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3,
                padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.conv_res = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn_res = nn.BatchNorm2d(out_channels)
            self.maxpool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()

        def forward(self, x):

            residual = self.conv_res(x)
            residual = self.bn_res(residual)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)

            x += residual
            x = self.relu(x)
            x = self.maxpool(x)

            return x

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = torch.nn.Sequential(
            self.Block(3, 64),
            self.Block(64, 128),
            self.Block(128, 256),
            self.Block(256, 512),
            torch.nn.AvgPool2d(int(image_size / 16), stride=1),
            Flatten(),
            torch.nn.Linear(512, num_classes)
            )
        self.reset_parameters()

    def reset_parameters(self):

        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out',
                    nonlinearity='relu')

    def forward(self, x):

        x = self.model(x)
        return x




class WRN2810(torch.nn.Module):

    def __init__(self, num_classes, image_size):

        super().__init__()

        self.model = \
            partial(wrn, num_groups=3, N=4, k=10)(num_classes=num_classes)

    def forward(self, x):

        x = self.model(x)
        return x


class Normalize(torch.nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        if len(self.mean)>1:
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
        else:
            mean, std = self.mean, self.std
        return (input - mean) / std


def load_resnet_norm(num_classes, image_size):
    
    MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010] 
    # MEAN, STD = [0]*3, [1]*3


    base_model = ResNet12(num_classes, image_size)    
    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = torch.nn.Sequential(
        norm_layer,
        base_model)

    return model


def load_conv64_norm(num_classes, image_size):
    
    MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010] 
    # MEAN, STD = [0]*3, [1]*3


    base_model = Conv64(num_classes, image_size)
    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = torch.nn.Sequential(
        norm_layer,
        base_model)

    return model


def load_wrn_norm(num_classes, image_size):
    
    MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010] 
    # MEAN, STD = [0]*3, [1]*3


    base_model = WRN2810(num_classes, image_size)
    
    norm_layer = Normalize(mean=MEAN, std=STD)

    model = torch.nn.Sequential(
        norm_layer,
        base_model)

    return model



class MultiLPDataset(torch.utils.data.Dataset):
    """Dataset wrapper to have low-pass dataset"""

    def __init__(self, dataset, radius):
        
        self.dataset = dataset ## Recieve (Transformed) dataset
        self.radius = radius
        print(f'Generating Low Pass Images for : {self.radius}')

        self.get_lp()

    
    def get_lp(self):

        self.lp = [[] for _ in range(len(self.dataset))]*(len(self.dataset))

        for idx in tqdm(range(len(self.dataset)), total=len(self.dataset), leave=False):
            x, _ = self.dataset[idx]
            x = x.unsqueeze(0)

            for r in self.radius:
                lp_x, _ = get_freq(x, r)

                lp_x = lp_x.squeeze(0)
                self.lp[idx].append(lp_x)

    def __getitem__(self, i):
        x, y = self.dataset[i] ## Original Sample
        lp_x = self.lp[i]
        return (x, lp_x, y)
        
    def __len__(self):
        return len(self.dataset)




class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

