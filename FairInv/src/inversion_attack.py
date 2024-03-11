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

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from . import utils
from . import models
from . import os_layer

torch.manual_seed(1)
np.random.seed(7)
sns.set(style="white", palette="muted", color_codes=True, context="talk")
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('grid', linestyle="dotted", color='black')


def main(args: argparse.Namespace, log: logging.Logger) -> None:

    raw_path: Path = Path(args.raw_path)
    raw_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(raw_path, log)
    if raw_path_status is None:
        msg: str = f"Something went wrong when creating {raw_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)


    X, y, Z = utils.load_ICU_data('data/adult.data')
    X = SelectKBest(f_classif, k=10).fit_transform(X, y)
    X = pd.DataFrame(X)

    n_features = X.shape[1]
    n_sensitive = Z.shape[1]

    # split into train/test set
    (X_train, X_test, y_train, y_test,Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5,stratify=y, random_state=7)
    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler) 
    # get training data for adversary
    (X_adv_train, X_adv_test, y_adv_train, y_adv_test, Z_adv_train, Z_adv_test) = train_test_split(X_test, y_test, Z_test, test_size=0.5,stratify=y_test, random_state=7)


    train_data = utils.PandasDataSet(X_train, y_train, Z_train)
    test_data = utils.PandasDataSet(X_test, y_test, Z_test)
    adv_train_data = utils.PandasDataSet(X_adv_train, y_adv_train, Z_adv_train)
    adv_test_data = utils.PandasDataSet(X_adv_test, y_adv_test, Z_adv_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)
    print('Target Model Data: # training samples: {}; # testing samples: {}'.format(len(train_data),len(test_data)))
    print('Adversary Data: # training samples: {}; # testing samples: {}'.format(len(adv_train_data),len(adv_test_data)))

    clf = models.Classifier(n_features=n_features)
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clf.parameters())

    for epoch in range(args.n_clf_epochs):
        # print("Training Classifier: {}".format(epoch))
        clf = utils.pretrain_classifier(clf, train_loader, test_loader, clf_optimizer, clf_criterion)

    lambdas = torch.Tensor([130, 30])
    adv = models.Adversary(Z_train.shape[1])
    adv_criterion = nn.BCELoss(reduce=False)
    adv_optimizer = optim.Adam(adv.parameters())

    for epoch in range(args.n_adv_epochs):
        # print("Training Adversary Model:", epoch)
        utils.pretrain_adversary(adv, clf, train_loader, adv_optimizer, adv_criterion, lambdas)

    with torch.no_grad():
        pre_clf_test = clf(test_data.tensors[0])
        pre_adv_test = adv(pre_clf_test)


    y_pre_clf = pd.Series(pre_clf_test.data.numpy().ravel(),index=y_test.index)
    y_pre_adv = pd.DataFrame(pre_adv_test.numpy(), columns=Z.columns)
    Z_pre_adv = pd.DataFrame(pre_adv_test.numpy(), columns=Z_test.columns)
    # plot for non-fair models
    fig = utils.plot_distributions(y_test, Z_test, y_pre_clf, y_pre_adv) 
    fig.savefig('./pre_training_dist.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)

    print("Normal Training Classifier Accuracy: Train: {:.2f}; Test: {:.2f}".format(utils.test_clf(clf,train_loader)*100,utils.test_clf(clf,test_loader)*100))
    p_rules = {'race': utils.p_rule(y_pre_clf, Z_test['race']),'sex': utils.p_rule(y_pre_clf, Z_test['sex'])}
    print("Normal Training: Fairness Metric: race: {:.2f}; sex: {:.2f}".format(p_rules['race'], p_rules['sex']))
    print("Normal Training: Adversary Performance (AUCROC): {:.2f}".format(metrics.roc_auc_score(Z_test, Z_pre_adv)))

    ################### Attack Normal Model ########################
    with torch.no_grad():
        adv_clf_pred_train = clf(adv_train_data.tensors[0])
        adv_clf_pred_test = clf(adv_test_data.tensors[0])
    y_pred_adv_train = pd.Series(adv_clf_pred_train.numpy().ravel(), index=y_adv_test.index)
    y_pred_adv_test = pd.Series(adv_clf_pred_test.numpy().ravel(), index=y_adv_test.index)

    normal_model_runs = []
    for _ in range(10):
        test_loss = utils.inversion_attack(X_adv_train, X_adv_test, y_pred_adv_train, y_pred_adv_test)
        normal_model_runs.append(test_loss)
    normal_model_runs = np.array(normal_model_runs)
    print("Loss for Normal Model: mean: {}; std: {}".format(np.mean(normal_model_runs), np.std(normal_model_runs)))


    ####################### Fair Training #########################


    for epoch in range(1, args.n_epoch_combined):
        if epoch//50==0:
            print("Fair Training Epoch:", epoch)
        clf, adv = utils.train(clf, adv, train_loader, clf_criterion, adv_criterion,clf_optimizer, adv_optimizer, lambdas)

        with torch.no_grad():
            clf_pred = clf(test_data.tensors[0])
            adv_pred = adv(clf_pred)

        y_post_clf = pd.Series(clf_pred.numpy().ravel(), index=y_test.index)
        Z_post_adv = pd.DataFrame(adv_pred.numpy(), columns=Z_test.columns)


    # plot for non-fair models
    fig = utils.plot_distributions(y_test, Z_test, y_post_clf, Z_post_adv, epoch)
    # display.clear_output(wait=True)
    plt.savefig(f'post_training_dist.pdf', bbox_inches = 'tight',pad_inches = 0, dpi=400)

    print("Post Fair Training Classifier Accuracy: Train: {:.2f}; Test: {:.2f}".format(utils.test_clf(clf,train_loader)*100,utils.test_clf(clf,test_loader)*100))
    post_p_rules= {'race': utils.p_rule(y_post_clf, Z_test['race']),'sex': utils.p_rule(y_post_clf, Z_test['sex'])}
    print("Post Fair Training: Fairness Metric: race: {:.2f}; sex: {:.2f}".format(post_p_rules['race'], post_p_rules['sex']))
    print("Post Fair Training: Adversary Performance (AUCROC): {:.2f}".format(metrics.roc_auc_score(Z_test, Z_post_adv)))

    ################### Attack Fair Model ########################
    with torch.no_grad():
        adv_clf_pred_train = clf(adv_train_data.tensors[0])
        adv_clf_pred_test = clf(adv_test_data.tensors[0])
    y_pred_adv_train = pd.Series(adv_clf_pred_train.numpy().ravel(), index=y_adv_test.index)
    y_pred_adv_test = pd.Series(adv_clf_pred_test.numpy().ravel(), index=y_adv_test.index)

    fair_model_runs = []
    for _ in range(10):
        test_loss = utils.inversion_attack(X_adv_train, X_adv_test, y_pred_adv_train, y_pred_adv_test)
        fair_model_runs.append(test_loss)
    fair_model_runs = np.array(fair_model_runs)
    print("Loss for Fair Model: mean: {}; std: {}".format(np.mean(fair_model_runs), np.std(fair_model_runs)))

    print("Statistical Signficance Test: {}".format(stats.ttest_ind(normal_model_runs, fair_model_runs)))

def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="data/", help='Root directory of the dataset')
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning Rate")
    parser.add_argument("--decay", type = float, default = 0, help = "Weight decay/L2 Regularization")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training data")
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument("--save", type = bool, default = True, help = "Save model")
    parser.add_argument("--n_adv_epochs", type = int, default = 5, help = "Number of Adversary Model Training Iterations")
    parser.add_argument("--n_clf_epochs", type = int, default = 10, help = "Number of Classifier Model Training Iterations")
    parser.add_argument("--n_epoch_combined", type = int, default = 165, help = "Number of Classifier+Adversary Model Training Iterations")

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="inversion_attack.log", filemode="w")
    log: logging.Logger = logging.getLogger("InversionAttack")
    args = handle_args()
    main(args, log)