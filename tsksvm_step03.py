#!/usr/bin/python2.7
#tsksvm_step01: PREPROCESS AA SCALES
#tsksvm_step02: DEFAULT 'SELECT MODEL' PARAMETERS
from tsksvm_step02 import *

# PREPROCESS DATASETS

# 1 Training Dataset

# 1.1 Load training data file

# input: training file
# output: 1. numPeptides (numeric values of peptides) 2. ytrain (response value of peptide) (peptide binding affinity)
train_file = "data/task1trnpep.txt"
numPeptides, ytrain = tsksvm.load_dataset(train_file)

# input: numPeptides, scales
# output: trnin dataset as scales
trnin = tsksvm.peptide2scales(numPeptides,scales)

# 2 Testing Dataset

# 2.1 Load testing data file

# input: testing file
# output: 1. numPeptides (numeric values of peptides) 2. ytest (response value of peptide) (peptide binding affinity)
test_file = "data/task1tstpep.txt"
numPeptides, ytest = tsksvm.load_dataset(test_file)

# input: numPeptides, scales
# output: chkin dataset as scales
chkin = tsksvm.peptide2scales(numPeptides,scales)

