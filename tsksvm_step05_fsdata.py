#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS
#tsksvm_step04_mcfs_ml: FEATURE SELECTION
from tsksvm_step04_mcfs_ml import *

# 1 Transform Training Dataset with Selected Features

Xtrain = trnin.as_matrix()[:,selfea.tolist()]
Xtrain = pd.DataFrame(Xtrain)

# 2 Transform Testing Dataset with Selected Features

Xtest = chkin.as_matrix()[:,selfea.tolist()]
Xtest = pd.DataFrame(Xtest)
