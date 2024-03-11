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
from tqdm import tqdm
import os
import torch
import torch.utils.data as Data

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from . import data
from . import models
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

        dataset = data.CensusWrapper(path=raw_path,filter_prop=args.filter, ratio=float(args.ratio), split=args.split)
        (X_train, y_train, Z_train), (X_test, y_test, Z_test) = dataset.load_data()
  
        traindata = Data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor))
        testdata = Data.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor))
        trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=64, shuffle=False)
        testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=64, shuffle=False)


        for i in range(args.num):
            model = models.BinaryNet(X_train.shape[1]).to(args.device)
            model, trainacc, testacc = utils.train(args.epochs, model, trainloader, testloader, torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay), args, log)
  
            save_path = result_path / args.split / args.filter / str(args.ratio)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path / str(str(i) + "_%.2f" % testacc))

    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="dataset/", help='Root directory of the dataset')
    parser.add_argument('--result_path', type=str, default="results/", help='Results directory for the models')
    parser.add_argument('--dataset', type=str, default="CENSUS", help='Options: CENSUS, BONEAGE, CELEBA, ARXIV')
    parser.add_argument('--filter', type=str,required=False,help='while filter to use')
    parser.add_argument('--ratio', type=float, default=0.5,help='what ratio of the new sampled dataset')
    parser.add_argument('--num', type=int, default=10,help='how many classifiers to train?')
    parser.add_argument('--split', choices=["attacker", "victim"],required=True,help='which split of data to use')
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="generate_models.log", filemode="w")
    log: logging.Logger = logging.getLogger("GenerateModels")
    args = handle_args()
    main(args, log)
