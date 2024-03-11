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

import copy
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


class Measure(object):

    def __init__(self):
        self.name = 'None'

    def __str__(self):
        return self.name

    def restart_model(self, X_train, y_train, model):
        try:
            model = copy.deepcopy(model)
        except:
            model.fit(np.zeros((0,) + X_train.shape[1:]), y_train)

class Mem(Measure):

    def __init__(self):
        self.name = 'Mem'

    def score(self, X_train, y_train, X_test, y_test, model=None):

        sources = {i:np.array([i]) for i in range(X_train.shape[0])}
        self.restart_model(X_train, y_train, model)
        model.fit(X_train, y_train)
        vals_loo = np.zeros(X_train.shape[0])
        for i in sources.keys():
            deleted_x = np.expand_dims(X_train[i], axis=0)
            deleted_y = y_train[i]
            baseline_value = model.predict_proba(deleted_x)
            # print(baseline_value)
            X_batch = np.delete(X_train, sources[i], axis=0)
            y_batch = np.delete(y_train, sources[i], axis=0)
            model.fit(X_batch, y_batch)
            removed_value = model.predict_proba(deleted_x)
            vals_loo[sources[i]] = (round(baseline_value[0][deleted_y],2) - round(removed_value[0][deleted_y],2))/len(sources[i])
        return vals_loo

def plot_fig(X, clf,overfitting,mem,case):
    figure = plt.figure(figsize=(4, 3))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # cm = plt.cm.RdBu
    cm = ListedColormap(["#e3af88", "#babaf7"])
    cm_bright = ListedColormap(["#d48448", "#5656d1"])

    ax = plt.subplot(1,1,1)

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max] x [y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0],X_train[:, 1],c=y_train,cmap=cm_bright,edgecolors="black",s=45)
    ax.scatter(X_test[:, 0],X_test[:, 1],c=y_test,cmap=cm_bright,edgecolors="black",marker='P',s=45)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.text(xx.max() - 0.1,yy.min() + 0.2,f"Overfitting: {overfitting:.2f}".lstrip("0"),size=10,horizontalalignment="right")
    ax.text(xx.min() + 0.1,yy.min() + 0.2,f"Mem: {mem:.2f}".lstrip("0"),size=10)
    # plt.show()
    plt.savefig("./{}.pdf".format(case), bbox_inches = 'tight',pad_inches = 0, dpi=400)


# Case1 : No overfitting and no memorization (dataset is linearly seperable and LOO stable)
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, n_features=2,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf_case1 = make_pipeline(StandardScaler(),MLPClassifier(solver="lbfgs",alpha=0,random_state=1,max_iter=2000,early_stopping=True,hidden_layer_sizes=[10, 10]))
clf_case1.fit(X_train, y_train)
train_score = clf_case1.score(X_train, y_train)
test_score = clf_case1.score(X_test, y_test)
overfitting = train_score - test_score
mem = Mem()
memscores = mem.score(X_train, y_train, X_test, y_test, clf_case1)
plot_fig(X, clf_case1, overfitting,memscores.mean(),"case1")

# Case 2 : Overfitting and no memorization
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, n_features=2,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
i_0=0
i_1=0
for x,y in zip(X_test, y_test):
    if y == 0 and i_0<=50:
        x[1] = x[1] - random.choice([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2])
        i_0+=1
    if y==1 and i_1<=50:
        x[1] = x[1] + random.choice([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2])
        i_1+=1
clf_case2 = make_pipeline(StandardScaler(),MLPClassifier(solver="lbfgs",alpha=0,random_state=1,max_iter=2000,early_stopping=True,hidden_layer_sizes=[10, 10]))
clf_case2.fit(X_train, y_train)
train_score = clf_case2.score(X_train, y_train)
test_score = clf_case2.score(X_test, y_test)
overfitting = train_score - test_score
X = np.concatenate((X_train, X_test))
Y = np.concatenate((y_train, y_test))
mem = Mem()
memscores = mem.score(X_train, y_train, X_test, y_test, clf_case2)
plot_fig(X, clf_case2, overfitting,memscores.mean(),"case2")


# Case 3 : No overfitting but memorization
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, n_features=2,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
for x,y in zip(X_train, y_train):
    if y == 0 and i_0<=50:
        x[1] = x[1] - random.choice([0,0.25,0.5,0.75,1,1.25,1.5])
        i_0+=1
    if y==1 and i_1<=50:
        x[1] = x[1] + random.choice([0,0.25,0.5,0.75,1,1.25,1.5])
        i_1+=1
clf_case3 = make_pipeline(StandardScaler(),MLPClassifier(solver="lbfgs",alpha=0,random_state=1,max_iter=2000,early_stopping=True,hidden_layer_sizes=[10, 10]))
clf_case3.fit(X_train, y_train)
train_score = clf_case3.score(X_train, y_train)
test_score = clf_case3.score(X_test, y_test)
overfitting = train_score - test_score
mem = Mem()
memscores = mem.score(X_train, y_train, X_test, y_test, clf_case3)
plot_fig(X, clf_case3, overfitting,memscores.mean(),"case3")

# Case 4: Overfitting and Memorization.
X, y = make_moons(n_samples=100,noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf_case4 = make_pipeline(StandardScaler(),MLPClassifier(solver="lbfgs",alpha=0,random_state=1,max_iter=2000,early_stopping=True,hidden_layer_sizes=[10, 10]))
clf_case4.fit(X_train, y_train)
train_score = clf_case4.score(X_train, y_train)
test_score = clf_case4.score(X_test, y_test)
overfitting = train_score - test_score
mem = Mem()
memscores = mem.score(X_train, y_train, X_test, y_test, clf_case4)
plot_fig(X, clf_case4, overfitting,memscores.mean(),"case4")
