#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
from tsksvm_step01 import *

# DEFAULT SELECT MODEL PARAMETERS

# init: cnum=2, svrc=1.0, svrp=0.05
sm = tsksvm.SelectModel(cnum=2, svrc=1.0, svrp=0.05)