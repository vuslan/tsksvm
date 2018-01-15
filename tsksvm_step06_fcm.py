#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS
#tsksvm_step04_mcfs_ml: FEATURE SELECTION
#tsksvm_step05_fsdata: TRANSFORM DATASETS BASED ON FEATURE SELECTION
from tsksvm_step05_fsdata import *

# FUZZY C MEANS CLUSTERING

M, inpstd = sm.identify_fuzzy_system_with_fcm(Xtrain, ytrain, sm.cnum)