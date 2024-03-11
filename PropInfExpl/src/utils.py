# Authors: Vasisht Duddu
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
import torch
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from copy import deepcopy
import numpy as np
from sklearn.neural_network import MLPClassifier
from captum.attr import DeepLift,IntegratedGradients,NoiseTunnel

from . import models

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('grid', linestyle="dotted", color='black')

class AverageMeter(object):
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

def train(epochs, model, trainloader, testloader, optimizer, args, log):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #loss = nn.NLLLoss(output,target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')

    trainacc = test(model, trainloader, args, log)
    testacc = test(model, testloader, args, log)
    return model, trainacc, testacc


def test(model, loader, args, log):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(loader.sampler)
        accuracy = 100. * correct / len(loader.sampler)
        print("Accuracy: {}/{} ({}%)".format(correct,len(loader.sampler),accuracy))
    return accuracy


def train_boneage(args, model, loaders, lr=1e-3, epoch_num=10,weight_decay=0, get_best=False):
    train_loader, val_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

    iterator = tqdm(range(1, epoch_num+1))
    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(args, train_loader, model,criterion, optimizer, epoch)
        vloss, vacc = validate_epoch(args, val_loader, model, criterion)
        iterator.set_description("train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))
        if get_best and vloss < best_loss:
            best_loss = vloss
            best_model = deepcopy(model)
    if get_best:
        return best_model, (vloss, vacc)
    return vloss, vacc

def train_epoch(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    iterator = tqdm(train_loader)
    for data in iterator:
        images, labels = data
        images, labels = images.to(args.device), labels.to(args.device)
        N = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)[:, 0]

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        prediction = (outputs >= 0)
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(args, val_loader, model, criterion):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            N = images.size(0)
            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)
            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
            val_loss.update(criterion(outputs, labels.float()).item())

    print('[Validation], Loss: %.5f, Accuracy: %.4f' %(val_loss.avg, val_acc.avg))
    return val_loss.avg, val_acc.avg

def get_model_list(args, load_path_ratio1, load_path_ratio2, test_load_path_ratio1, test_load_path_ratio2, shape):
    model_list_ratio1 = []
    for suffix in os.listdir(load_path_ratio1):
        model_path = load_path_ratio1 / suffix
        model = models.BinaryNet(shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_list_ratio1.append(model)

    model_list_ratio2 = []
    for suffix in os.listdir(load_path_ratio2):
        model_path = load_path_ratio2 / suffix
        model = models.BinaryNet(shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_list_ratio2.append(model)

    test_model_list_ratio1 = []
    for suffix in os.listdir(test_load_path_ratio1):
        model_path = test_load_path_ratio1 / suffix
        model = models.BinaryNet(shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_model_list_ratio1.append(model)

    test_model_list_ratio2 = []
    for suffix in os.listdir(test_load_path_ratio2):
        model_path = test_load_path_ratio2 / suffix
        model = models.BinaryNet(shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_model_list_ratio2.append(model)

    return model_list_ratio1, model_list_ratio2, test_model_list_ratio1, test_model_list_ratio2


def get_model_list_capacity_exp(args, load_path_ratio1, load_path_ratio2, test_load_path_ratio1, test_load_path_ratio2, shape):


    def get_model(args, shape):
        if args.model_number == 1:
            return models.Model1(shape)
        elif args.model_number == 2:
            return models.Model2(shape)
        elif args.model_number == 3:
            return models.Model3(shape)
        else:
            return models.BinaryNet(shape)

    model_list_ratio1 = []
    for suffix in os.listdir(load_path_ratio1):
        model_path = load_path_ratio1 / suffix
        model = get_model(args, shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_list_ratio1.append(model)

    model_list_ratio2 = []
    for suffix in os.listdir(load_path_ratio2):
        model_path = load_path_ratio2 / suffix
        model = get_model(args, shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_list_ratio2.append(model)

    test_model_list_ratio1 = []
    for suffix in os.listdir(test_load_path_ratio1):
        model_path = test_load_path_ratio1 / suffix
        model = get_model(args, shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_model_list_ratio1.append(model)

    test_model_list_ratio2 = []
    for suffix in os.listdir(test_load_path_ratio2):
        model_path = test_load_path_ratio2 / suffix
        model = get_model(args, shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_model_list_ratio2.append(model)

    return model_list_ratio1, model_list_ratio2, test_model_list_ratio1, test_model_list_ratio2

def generate_explanations(args,model_list,input,baseline):

    attribution_list = []
    delta_list = []
    explanations_list = []

    for model in model_list:
        if args.explanations == "IntegratedGradients":
            ig = IntegratedGradients(model)
            attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)

        elif args.explanations == "DeepLift":
            dl = DeepLift(model)
            attributions, delta = dl.attribute(input, baseline, target=0, return_convergence_delta=True)

        # elif args.explanations == "GradientShap":
        #     gs = GradientShap(model)
        #     baseline_dist_train = torch.randn(input.size()) * 0.001
        #     attributions, delta = gs.attribute(input, stdevs=0.09, n_samples=4, baselines=baseline_dist_train,target=0, return_convergence_delta=True)
        #     delta = torch.mean(delta.reshape(input.shape[0], -1), dim=1)

        elif args.explanations == "smoothgrad":
            ig = IntegratedGradients(model)
            nt = NoiseTunnel(ig)
            attributions, delta = nt.attribute(input, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,baselines=baseline, target=0, return_convergence_delta=True)
            delta = torch.mean(delta.reshape(input.shape[0], -1), dim=1)

        else:
            msg_algo_error: str = "No such algorithm"
            log.error(msg_algo_error)
            raise EnvironmentError(msg_algo_error)

        explanations = np.concatenate((attributions.detach().numpy(), np.expand_dims(delta.numpy(), axis=0).T), axis=1)

        attribution_list.append(attributions.detach().numpy())
        delta_list.append(delta.detach().numpy())
        explanations_list.append(explanations)
    
    attribution_array = np.empty(attribution_list[0].shape)
    for arr in attribution_list:
        attribution_array = np.concatenate((attribution_array, arr), axis=0)

    delta_array = np.empty(delta_list[0].shape)
    for arr in delta_list:
        delta_array = np.concatenate((delta_array, arr), axis=0)    

    explanation_array = np.empty(explanations_list[0].shape)
    for arr in explanations_list:
        explanation_array = np.concatenate((explanation_array, arr), axis=0)       

    return attribution_array, delta_array, explanation_array

def propinfattack(X_attributions_train, y_train, X_attributions_test, y_test):

    df_train = pd.DataFrame()
    df_train = pd.concat([df_train,pd.DataFrame(X_attributions_train)])
    df_train['Y'] = y_train.tolist()
    df_train = df_train.dropna()
    y_train = df_train['Y'].to_numpy()
    X_attributions_train = df_train.drop(['Y'], axis=1).to_numpy()

    df_test = pd.DataFrame()
    df_test = pd.concat([df_test,pd.DataFrame(X_attributions_test)])
    df_test['Y'] = y_test.tolist()
    df_test = df_test.dropna()
    y_test = df_test['Y'].to_numpy()
    X_attributions_test = df_test.drop(['Y'], axis=1).to_numpy()

    model = MLPClassifier(hidden_layer_sizes=(512,256,128,64,2), solver ='adam', max_iter=1000).fit(X_attributions_train, y_train)
    print("Accuracy with Attributions: {:.2f}".format(model.score(X_attributions_test, y_test)*100))
    return model.score(X_attributions_test, y_test)*100
