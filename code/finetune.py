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

import os
import torchattacks
from utils import *

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

def get_args():

    parser = argparse.ArgumentParser(description='Finetune',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='Mini-ImageNet')
    parser.add_argument('--architecture', type=str, default='WRN2810')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--support_shot', type=int, default=1)
    parser.add_argument('--query_shot', type=int, default=15)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--train_and_val', action='store_true')
    parser.add_argument('--non_transductive', action='store_true')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--attack', default='pgd', type=str, help='Which Attack to use')
    parser.add_argument('--reg', default=None, help='Regularization Method to Use', choices=['cosine_lp_q', 'cosine_lp_s', 'cosine_lp_sq'])
    parser.add_argument('--use_last', action='store_true')
    parser.add_argument('--mode', type=str, default='fixed', choices=['uniform', 'curr'])
    parser.add_argument('--r_max', default=8, type=int, help='Radius to use for Cosine LP Loss (Uniform Distribution)')
    parser.add_argument('--run_name', default=None, type=str, help='Run Name for the Exp.')

    args = parser.parse_args()

    if args.non_transductive:
        assert args.reg == 'cosine_lp_s' ## Can only use support samples

    if args.gpu:
        print(f'Using GPU: {args.gpu}')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.cuda.set_device(int(args.gpu))


    return args.dataset, args.architecture, args.way, args.support_shot, \
        args.query_shot, args.num_episodes, args.train_and_val, \
        args.non_transductive, args

def fine_tune(backbone, relu, lamb, support, query, non_transductive, args):

    '''
    Transductive Fine-Tuning
    The idea is to use information from the test datum to restrict the
    hypothesis space while searching for the classifier at test time. We
    introduce a regularizer on the test data as we seek outputs with a peaked
    posterior, or low Shannon Entropy.
    '''

    lr = 5e-5
    epochs = 25

    model = FewShotModel(backbone, support, relu).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # hardness = hardness_model.get_hardness(support, query)

    init_accuracy = validate(model, query)

    model.train()
    training_times = []

    for epoch in range(1, epochs + 1):
        
        start_epoch = time.time()
        
        for (xs, ys), (xq, yq) in zip(support.train_dl, query.train_dl):
            break
        micro_forward(model, xs, ys, cross_entropy, 1.)
        optimizer.step()
        if not non_transductive:
            micro_forward(model, xq, yq, entropy, lamb)
            optimizer.step()

        ## Apply Cosine LP Loss (Regularization)
        if args.reg == 'cosine_lp_q': ## Only Using Query Samples
            micro_forward_reg(model, xq, args=args)
            optimizer.step()

        elif args.reg == 'cosine_lp_s': ## Only Using Support Samples
            # print(f'Using Support Samples for Loss: {xs.shape}')
            micro_forward_reg(model, xs, args=args)
            optimizer.step()

        elif args.reg == 'cosine_lp_sq': ## Using Support + Query Samples
            x_sq = torch.cat((xs, xq), dim=0)
            micro_forward_reg(model, x_sq, args=args)
            optimizer.step()

        end_epoch = time.time()
        training_elapsed = end_epoch - start_epoch
        training_times.append(training_elapsed)

    final_accuracy = validate(model, query)
    if args.attack == 'auto_attack':
            attack = torchattacks.AutoAttack(model, eps=8/255, n_classes=args.way, version='standard')
    elif args.attack == 'pgd':
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)

    final_accuracy_adv = validate_adv(model, query, attack, args)

    clean_ensemble_acc, adv_ensemble_acc = validate_ensemble(model, query, attack, radiuses=args.ensemble_rad, args=args)

    return init_accuracy, final_accuracy, final_accuracy_adv, clean_ensemble_acc, adv_ensemble_acc

def main():

    seed_everything(42)

    dataset, architecture, way, support_shot, query_shot, num_episodes, \
        train_and_val, non_transductive, args = get_args()
    params = get_params(dataset, architecture, train_and_val)
    relu = params['relu']
    lamb = params['lambda']
    image_size = params['image_size']
    csv_file_name = params['csv_file_name']

    if args.run_name is None:
        parameter_path = os.path.join(dataset, 'models', params['parameter_file_name'] + '_mixup.pth')
    else:
        if args.use_last: 
            parameter_path = os.path.join(dataset, 'models', args.run_name + '_' + params['parameter_file_name'] + '_student_last.pth')
        else:
            parameter_path = os.path.join(dataset, 'models', args.run_name + '_' + params['parameter_file_name'] + '_student.pth')
    
 
    data = FewShotDataset(dataset, 'test.csv', image_size, way, support_shot,
        query_shot)

    meta_train_data = get_data(dataset, csv_file_name, csv_file_name,
        image_size, 1)
    meta_train_classes = len(meta_train_data.valid_ds.y.classes)
    backbone = globals()[architecture](num_classes=meta_train_classes,
        image_size=image_size)

    if args.architecture=='ResNet12':
        backbone = load_resnet_norm(num_classes=meta_train_classes, image_size=image_size,)
    elif args.architecture=='conv64':    
        backbone = load_conv64_norm(num_classes=meta_train_classes, image_size=image_size,)
    elif args.architecture=='WRN2810':    
        backbone = load_wrn_norm(num_classes=meta_train_classes, image_size=image_size,)
    else:
        raise NotImplementedError

    state_dict = torch.load(parameter_path, map_location='cpu')
    backbone.load_state_dict(state_dict['net'])
    backbone = backbone.cuda()

    print(f'Run : {args.run_name} | Model Loaded : {parameter_path} | Acc : {state_dict["acc"]} | Best Epoch: {state_dict["epoch"]}')
    print(f'Evaluating on {args.attack}, Using {args.reg}')


    ## Ensembling Utilities
    try:
        args.ensemble_rad = state_dict['running_radiuses']
    except:
        args.ensemble_rad = list(range(args.r_max, 1, -1))
    
    print(f'Radius Range for Ensembling: {args.ensemble_rad}')

    if args.mode == 'uniform':
        print(f'Radius For Training: {args.r_max} - 2')
        args.weights = [1]*len(args.ensemble_rad) ## For uniform sampling
        print(f'Wieghts for finetuning: {args.weights}')
    elif args.mode == 'curr':
        print(f'Radius For Training: {args.ensemble_rad[0]} - {args.ensemble_rad[-1]}')
        weights = state_dict['weight_dist']
        args.weights = weights
        formatted_w = [ '%.2f' % elem for elem in args.weights]
        print(f'Wieghts for finetuning: {formatted_w}')

    # hardness_model = Hardness()
    print(f'-'*100)

    results = {
        'dataset'           : dataset,
        'architecture'      : architecture,
        'way'               : way,
        'support_shot'      : support_shot,
        'query_shot'        : query_shot,
        'train_and_val'     : train_and_val,
        'non_transductive'  : non_transductive,
        'init_accuracy'     : [],
        'final_accuracy'    : [],
        'final_accuracy_adv': [],
        'clean_ensemble_acc': [],
        'adv_ensemble_acc'  : []
        }
    
    if args.run_name is None:
        file_name = 'teacher_' + dataset + '_' + architecture + '_' + str(way) + '_' \
            + str(support_shot ) + '_' + str(query_shot) + '_' \
            + str(train_and_val) + '_' + str(non_transductive)
    else:
        file_name = args.run_name + '_' + str(args.reg) + '_' + dataset + '_' + architecture + '_' + str(way) + '_' \
            + str(support_shot ) + '_' + str(query_shot) + '_' \
            + str(train_and_val) + '_' + str(non_transductive)
    os.makedirs('results', exist_ok=True)
    print(args)
    print(f'-'*100)

    pbar = tqdm(range(1, num_episodes + 1), leave=False, ncols=150)

    for episode in pbar:
        support, query = data[episode]
        init_accuracy, final_accuracy, final_accuracy_adv, clean_ensemble_acc, adv_ensemble_acc = fine_tune(backbone, relu,
            lamb, support, query, non_transductive, args)
        results['init_accuracy'].append(init_accuracy)
        results['final_accuracy'].append(final_accuracy)
        results['final_accuracy_adv'].append(final_accuracy_adv)
        results['clean_ensemble_acc'].append(clean_ensemble_acc)
        results['adv_ensemble_acc'].append(adv_ensemble_acc)

        pbar.set_description(f'Clean: {np.mean(results["final_accuracy"]):.2f} | Adv: {np.mean(results["final_accuracy_adv"]):.2f} \t Clean (Ensemble): {np.mean(results["clean_ensemble_acc"]):.2f} | Adv (Ensemble): {np.mean(results["adv_ensemble_acc"]):.2f}')

    torch.save(results, os.path.join('results', file_name + '.pth'))


    print('\t mean \t standard-deviation \t confidence-interval')
    print('Init Accuracy: ')
    print('\t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
        np.mean(results['init_accuracy']),
        np.std(results['init_accuracy']),
        1.96 * np.std(results['init_accuracy']) / (episode ** 0.5))
        )

    print('Final Accuracy') 
    print('\t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
        np.mean(results['final_accuracy']),
        np.std(results['final_accuracy']),
        1.96 * np.std(results['final_accuracy']) / (episode ** 0.5))
        )
    
    print('Final Adv. Accuracy') 
    print('\t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
        np.mean(results['final_accuracy_adv']),
        np.std(results['final_accuracy_adv']),
        1.96 * np.std(results['final_accuracy_adv']) / (episode ** 0.5))
        )
    
    print('Ensemble Clean Accuracy') 
    print('Final \t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
        np.mean(results['clean_ensemble_acc']),
        np.std(results['clean_ensemble_acc']),
        1.96 * np.std(results['clean_ensemble_acc']) / (episode ** 0.5))
        )

    print('Ensemble Adv Accuracy') 
    print('Final \t %2.2f \t\t %2.2f \t\t\t %2.2f' % (
        np.mean(results['adv_ensemble_acc']),
        np.std(results['adv_ensemble_acc']),
        1.96 * np.std(results['adv_ensemble_acc']) / (episode ** 0.5))
        )

if __name__ == '__main__':

    print('='*100)

    main()