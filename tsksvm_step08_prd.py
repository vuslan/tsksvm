#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS
#tsksvm_step04_mcfs_ml: FEATURE SELECTION
#tsksvm_step05_fsdata: TRANSFORM DATASETS BASED ON FEATURE SELECTION
#tsksvm_step06_fcm: FUZZY C MEANS CLUSTERING
#tsksvm_step07_trn: TRAINING FUZZYSVM
from tsksvm_step07_trn import *

# PREDICTING TRAIN DATASET

expout = ytrain.as_matrix()
prdout = tsksvm.predictfsvm(Xtrain,ytrain,M,inpstd,C,bias)
trnq2 = tsksvm.findq2(expout,prdout)


# PREDICTING TEST DATASET

expout = ytest.as_matrix()
prdout = tsksvm.predictfsvm(Xtest,ytest,M,inpstd,C,bias)
chkq2 = tsksvm.findq2(expout,prdout)


