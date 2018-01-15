#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS
#tsksvm_step04_mcfs_ml: FEATURE SELECTION
#tsksvm_step05_fsdata: TRANSFORM DATASETS BASED ON FEATURE SELECTION
#tsksvm_step06_fcm: FUZZY C MEANS CLUSTERING
#tsksvm_step07_trn: TRAINING FUZZYSVM
#tsksvm_step08_prd: PREDICTING TRAIN/TEST DATASET
from tsksvm_step08_prd import *

# RESULTS
print('trnq2: {} - chkq2:{}'.format(round(trnq2,2), round(chkq2,2)))
