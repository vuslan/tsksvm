#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS
#tsksvm_step04_mcfs_ml: FEATURE SELECTION
#tsksvm_step05_fsdata: TRANSFORM DATASETS BASED ON FEATURE SELECTION
#tsksvm_step06_fcm: FUZZY C MEANS CLUSTERING
from tsksvm_step06_fcm import *

# TRAINING FUZZYSVM

ker = 'linear'
clf, C, bias = tsksvm.trainfsvm(Xtrain, ytrain.values.ravel(), M, inpstd, ker, sm.svrc, sm.svrp)
