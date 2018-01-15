#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS

import pandas as pd
from tsksvm_step03 import *
from tsksvm_step04_mcfs_ml import *

# FEATURE SELECTION

# 1 Select Features

# 1.1 Selected feature set

from skfeature.utility import construct_W
kwargs_W = {"metric":"euclidean", "neighbor_mode":"knn", "weight_mode":"heat_kernel", "k":5, 't':1}
W = construct_W.construct_W(trnin, **kwargs_W)

num_fea = 161
from skfeature.function.sparse_learning_based import MCFS
score = MCFS.mcfs(trnin, num_fea, W=W)
idx = MCFS.feature_ranking(score)
selfea_new = idx[0:num_fea]

print("common features between selfea and selfea_new")
common=list(set(selfea[:num_fea]).intersection(selfea_new[:num_fea]))
