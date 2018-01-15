#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
#tsksvm_step03: PREPROCESS DATASETS

import pandas as pd
from tsksvm_step03 import *

# FEATURE SELECTION

# 1 Select Features

# 1.1 Selected feature set
selfea=pd.read_csv("task1env/selfea.dat",header=None)
selfea = selfea.transpose().values[0]-1
numInp=len(selfea)

