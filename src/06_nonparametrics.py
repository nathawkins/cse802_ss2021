import json
import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json
import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from parzen import ParzenWindowEstimate


# In[ ]:

parser = ArgumentParser()
parser.add_argument("-i", help = "Model index", type = int)
args = parser.parse_args()

def gaussian(x_):
    x = x_.sum()
    return 1/np.sqrt(2*np.pi) * np.exp(-1/2 * x**2)

models_ = {"KNN-7": KNeighborsClassifier(n_neighbors = 7),
           "KNN-9": KNeighborsClassifier(n_neighbors = 9),
           "KNN-11": KNeighborsClassifier(n_neighbors = 11),
           "KNN-13": KNeighborsClassifier(n_neighbors = 13),
           "KNN-15": KNeighborsClassifier(n_neighbors = 15),
           "KNN-17": KNeighborsClassifier(n_neighbors = 17),
           "KNN-19": KNeighborsClassifier(n_neighbors = 19),
           "KNN-21": KNeighborsClassifier(n_neighbors = 21),
           "Parzen-0.01": ParzenWindowEstimate(h = 0.01, kernel = gaussian),
           "Parzen-0.05": ParzenWindowEstimate(h = 0.05, kernel = gaussian),
           "Parzen-0.1": ParzenWindowEstimate(h = 0.1, kernel = gaussian),
           "Parzen-0.5": ParzenWindowEstimate(h = 0.5, kernel = gaussian),
           "Parzen-1": ParzenWindowEstimate(h = 1, kernel = gaussian),
           "Parzen-5": ParzenWindowEstimate(h = 5, kernel = gaussian),
           "Parzen-10": ParzenWindowEstimate(h = 10, kernel = gaussian),
           "Parzen-50": ParzenWindowEstimate(h = 50, kernel = gaussian)}




# In[ ]:


X = np.load("../data/features.npy")
y = np.load("../data/labels.npy")


# In[ ]:


results = {}
for k in models_.keys():
    results[k] = {"accuracy": 0.0, "auprc": 0.0, "f1": 0.0, "auroc": 0.0, "time": 0.0}


# In[ ]:

model_name = list(models_.keys())[args.i]
CLF = models_[model_name]

print(model_name)
print(CLF)


kf = KFold(n_splits = 5, shuffle = True, random_state = 8675309)
k = 0
for train_index, test_index in kf.split(X):
    k += 1
    if k > 3:
        break
    print(f"k = {k}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start_ = time()
    CLF.fit(X_train, y_train)
    y_pred = CLF.predict(X_test)
    end_ = time()

    print(end_ - start_)
        
    accuracy = accuracy_score(y_test, y_pred)
    auprc    = average_precision_score(y_test, y_pred)
    auroc    = roc_auc_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    
    results[model_name]["accuracy"] += accuracy
    results[model_name]["auprc"]    += auprc
    results[model_name]["auroc"]    += auroc
    results[model_name]["f1"]       += f1
    results[model_name]["time"]     += end_ - start_
        
for metric in results[model_name].keys():
    results[model_name][metric] /= 3


# In[ ]:

json.dump(results, open(f"../results/{k}_nonparametric_{'-'.join(model_name.split(' '))}.json", "w"))

