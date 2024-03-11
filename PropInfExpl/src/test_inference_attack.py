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
from typing import Optional
import os
import numpy as np
import torch


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from . import data
from . import os_layer
from . import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace, log: logging.Logger) -> None:

    raw_path: Path = Path(args.raw_path)
    raw_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(raw_path, log)
    result_path: Path = Path(args.result_path)
    result_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(result_path, log)
    if (raw_path_status or result_path_status) is None:
        msg: str = f"Something went wrong when creating {raw_path} and {result_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)
    log.info("Dataset {}".format(args.dataset))


    if args.dataset == "CENSUS":
        raw_path = raw_path / "census_splits"
        result_path = result_path / "CENSUS"
        dataset = data.CensusWrapper(path=raw_path,filter_prop=args.filter, ratio=float(args.ratio1), split="attacker")

        load_path_ratio1 = result_path / "attacker" / args.filter / str(args.ratio1)
        load_path_ratio2 = result_path / "attacker" / args.filter / str(args.ratio2)

        test_load_path_ratio1 = result_path / "victim" / args.filter / str(args.ratio1)
        test_load_path_ratio2 = result_path / "victim" / args.filter / str(args.ratio2)

    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)


    (_, _, _), (X_test_adv, y_test_adv, Z_test_adv) = dataset.load_data()
    indices = np.where(Z_test_adv==1)
    X_test_adv = X_test_adv[indices]
    y_test_adv = y_test_adv[indices]
    Z_test_adv = Z_test_adv[indices]

    model_list_ratio1, model_list_ratio2, test_model_list_ratio1, test_model_list_ratio2 = utils.get_model_list(args, load_path_ratio1, load_path_ratio2, test_load_path_ratio1, test_load_path_ratio2, X_test_adv.shape[1])

    input = torch.from_numpy(X_test_adv).type(torch.FloatTensor)
    baseline = torch.mean(input,dim=0)
    baseline = baseline.repeat(input.size()[0], 1)

    attributions_ratio1, _, _ = utils.generate_explanations(args,model_list_ratio1,input,baseline)
    attributions_ratio2, _, _ = utils.generate_explanations(args,model_list_ratio2,input,baseline)
    X_attributions_train = np.concatenate((attributions_ratio1, attributions_ratio2))
    y_train = np.concatenate((np.zeros(attributions_ratio1.shape[0]),np.ones(attributions_ratio2.shape[0])))

    test_attributions_ratio1, _, _ = utils.generate_explanations(args,test_model_list_ratio1,input,baseline)
    test_attributions_ratio2, _, _ = utils.generate_explanations(args,test_model_list_ratio2,input,baseline)
    X_attributions_test = np.concatenate((test_attributions_ratio1, test_attributions_ratio2))
    y_test = np.concatenate((np.zeros(test_attributions_ratio1.shape[0]),np.ones(test_attributions_ratio2.shape[0])))

    attacc_list = []
    for index in range(10):
        attacc = utils.propinfattack(X_attributions_train, y_train, X_attributions_test, y_test)
        attacc_list.append(attacc)
    attacc_list = np.array(attacc_list)
    print("##########################################################")
    print(attacc_list)
    def nan_if(arr, value):
        return np.where(arr == value, np.nan, arr)
    print("Attack Accuracy:{:.2f} $\\pm$ {:.2f}".format(np.nanmean(nan_if(attacc_list,50.00)),np.nanstd(nan_if(attacc_list,50.00))))

def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="dataset/", help='Root directory of the dataset')
    parser.add_argument('--result_path', type=str, default="results/", help='Results directory for the models')
    parser.add_argument('--dataset', type=str, default="CENSUS", help='Options: CENSUS, BONEAGE')
    parser.add_argument('--filter', type=str,required=False,help='while filter to use')
    parser.add_argument('--ratio1', type=float, default=0.5,help='what ratio of the new sampled dataset')
    parser.add_argument('--ratio2', type=float, default=0.5,help='what ratio of the new sampled dataset')
    parser.add_argument('--num', type=int, default=100,help='how many classifiers to train?')
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument('--explanations', type=str, default="IntegratedGradients", help='Options: IntegratedGradients,smoothgrad,DeepLift,GradientShap')

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="generate_models.log", filemode="w")
    log: logging.Logger = logging.getLogger("GenerateModels")
    args = handle_args()
    main(args, log)

