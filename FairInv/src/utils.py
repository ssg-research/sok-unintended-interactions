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

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import copy

from . import models

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def train(clf, adv, data_loader, clf_criterion, adv_criterion,clf_optimizer, adv_optimizer, lambdas):
    
    # Train adversary
    for x, y, z in data_loader:
        p_y = clf(x)
        adv.zero_grad()
        p_z = adv(p_y)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()
 
    # Train classifier on single batch
    for x, y, z in data_loader:
        pass
    p_y = clf(x)
    p_z = adv(p_y)
    clf.zero_grad()
    p_z = adv(p_y)
    loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
    clf_loss = clf_criterion(p_y, y) - (adv_criterion(adv(p_y), z) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()
    
    return clf, adv


def test_clf(model, test_loader):
    model.eval()
    output_list = []
    target_list = []
    with torch.no_grad():
        for data, target, _ in test_loader:
            output = model(data)
            output = output > 0.5
            output_list.append(output)
            target_list.append(target)
    output_list, target_list = np.array(np.concatenate(output_list)),np.array(np.concatenate(target_list))
    acc_score = accuracy_score(target_list, output_list)
    # print('Accuracy: {}'.format(acc_score*100))
    return acc_score

def pretrain_classifier(clf, data_loader, test_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y)
        loss.backward()
        optimizer.step()
    train_acc = test_clf(clf, data_loader)
    test_acc = test_clf(clf, test_loader)
    print("Train Acc: {:.2f}; Test Acc: {:.2f}".format(train_acc*100,test_acc*100))
    return clf


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion, lambdas):
    for x, _, z in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        p_z = adv(p_y)
        loss = (criterion(p_z, z) * lambdas).mean()
        loss.backward()
        optimizer.step()
    # train_acc = test_adv(clf, adv, data_loader)
    return adv

def load_ICU_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


def plot_distributions(y_true, Z_true, y_pred, Z_pred=None, epoch=None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    subplot_df = (
        Z_true
        .assign(race=lambda x: x['race'].map({1: 'white', 0: 'black'}))
        .assign(sex=lambda x: x['sex'].map({1: 'male', 0: 'female'}))
        .assign(y_pred=y_pred)
    )
    _subplot(subplot_df, 'race', ax=axes[0])
    _subplot(subplot_df, 'sex', ax=axes[1])
    # _performance_text(fig, y_true, Z_true, y_pred, Z_pred, epoch)
    fig.tight_layout()
    return fig


def _subplot(subplot_df, col, ax):
    for label, df in subplot_df.groupby(col):
        sns.kdeplot(df['y_pred'], ax=ax, label=label, fill=True)
    ax.set_title(f'Sensitive attribute: {col}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Output Distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(col))



def inversion_attack(X_adv_train, X_adv_test, y_pred_adv_train, y_pred_adv_test):


    train_data = PandasDataSet(y_pred_adv_train, X_adv_train)
    test_data = PandasDataSet(y_pred_adv_test, X_adv_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)

    model = models.Decoder(X_adv_train.shape[1])
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-2,weight_decay = 1e-4)

    outputs = []
    losses = []
    tmp_loss = 1e7
    for epoch in range(1000):
        for (latent_vector, output) in train_loader:
        
            reconstructed = model(latent_vector)
            loss = loss_function(reconstructed, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses.append(loss.detach().numpy())
        if loss.detach().numpy() < tmp_loss:
            best_model = copy.deepcopy(model)
            tmp_loss = loss.detach().numpy()
    
    # get loss on test dataset
    test_loss_array = []
    for (latent_vector, output) in test_loader:
        reconstructed = best_model(latent_vector)
        loss_test_function= torch.nn.MSELoss(reduction='mean')
        test_loss = loss_test_function(reconstructed, output)
        # print(test_loss)
        test_loss_array.append(test_loss.detach().numpy())

    return np.mean(np.array(test_loss_array))
    