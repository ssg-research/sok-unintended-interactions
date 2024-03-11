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

import torch.nn as nn

##### CENSUS #####
class BinaryNet(nn.Module):
    def __init__(self,num_features):
        super(BinaryNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 1024),nn.Tanh(),)
        self.fc2 = nn.Sequential(nn.Linear(1024, 512),nn.Tanh(),)
        self.fc3 = nn.Sequential(nn.Linear(512, 256),nn.Tanh(),)
        self.fc4 = nn.Sequential(nn.Linear(256, 128),nn.Tanh(),)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)




class Model1(nn.Module):
    def __init__(self,num_features):
        super(Model1, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 128),nn.Tanh(),)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        return self.classifier(out)



class Model2(nn.Module):
    def __init__(self,num_features):
        super(Model2, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 256),nn.Tanh(),)
        self.fc4 = nn.Sequential(nn.Linear(256, 128),nn.Tanh(),)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc4(out)
        return self.classifier(out)



class Model3(nn.Module):
    def __init__(self,num_features):
        super(Model3, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 256),nn.Tanh(),)
        self.fc2 = nn.Sequential(nn.Linear(256, 512),nn.Tanh(),)
        self.fc3 = nn.Sequential(nn.Linear(512, 256),nn.Tanh(),)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return self.classifier(out)


class Model4(nn.Module):
    def __init__(self,num_features):
        super(Model4, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 1024),nn.Tanh(),)
        self.fc2 = nn.Sequential(nn.Linear(1024, 512),nn.Tanh(),)
        self.fc3 = nn.Sequential(nn.Linear(512, 256),nn.Tanh(),)
        self.fc4 = nn.Sequential(nn.Linear(256, 128),nn.Tanh(),)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)



def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
