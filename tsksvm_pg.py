#!/usr/bin/python2.7
from sklearn.model_selection import ParameterGrid
import logging
import numpy as np
from tsksvm_step09_results import *

# REPEATPROCESS

# example 1
# grid = {'a': [1, 2], 'b': [True, False]}
# pg_list = list(ParameterGrid(grid))

# example 2
# grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
# pg_list = list(ParameterGrid(grid))

# https://www.blog.pythonlibrary.org/2016/05/03/python-201-an-intro-to-iterators-and-generators/
# http://www.diveintopython3.net/iterators.html
# http://anandology.com/python-practice-book/iterators.html

# param = {'cnum':7, 'svrc':2.40, 'svrp':0.05}
grid = [{'numInp': [160, 161], 'cnum': [x for x in range(2,3)], 'svrc': [1.00], 'svrp': [x for x in np.arange(0.5,0.6,0.1)]}]
pg_list = list(ParameterGrid(grid))

import pdb; pdb.set_trace()

opt = float("-inf")
L = []
df = pd.DataFrame(L)
df=pd.DataFrame(df, columns=['i', 'a/b', 'numInp', 'cnum', 'bias', 'svrc', 'svrp', 'trnq2', 'chkq2'])

i=1
for item in pg_list:

    sm.cnum = item['cnum']
    sm.svrc = item['svrc']
    sm.svrp = item['svrp']

    selfea = tsksvm.mcfs(trnin, item['numInp'])
    Xtrain = trnin.as_matrix()[:,selfea.tolist()]
    Xtrain = pd.DataFrame(Xtrain)
    Xtest = chkin.as_matrix()[:,selfea.tolist()]
    Xtest = pd.DataFrame(Xtest)

    M, inpstd = sm.identify_fuzzy_system_with_fcm(Xtrain, ytrain, sm.cnum)

    ker = 'linear'
    clf,C,bias = tsksvm.trainfsvm(Xtrain,ytrain.values.ravel(),M,inpstd,ker,sm.svrc,sm.svrp)
    sm.bias = float(bias)

    expout = ytrain.as_matrix()
    prdout = tsksvm.predictfsvm(Xtrain,ytrain,M,inpstd,C,bias)
    trnq2 = tsksvm.findq2(expout,prdout)
    sm.trnq2 = float(trnq2)

    expout = ytest.as_matrix()
    prdout = tsksvm.predictfsvm(Xtest,ytest,M,inpstd,C,bias)
    chkq2 = tsksvm.findq2(expout,prdout)
    sm.chkq2 = float(chkq2)
    logging.warn("i: {} - numInp: {} - cnum: {} - svrc: {} - svrp: {} - bias: {} - trnq2: {} - chkq2: {} - opt: {}".format(i, sm.numInp, sm.cnum, sm.svrc, sm.svrp, round(sm.bias,2), round(sm.trnq2,2), round(sm.chkq2,2), round(opt,2)))

    if sm.chkq2>opt:
       opt = sm.chkq2
       df.loc[i] = [round(i), 'a', round(item['numInp']), round(sm.cnum), round(sm.bias,2), sm.svrc, sm.svrp, round(sm.trnq2,2), round(sm.chkq2,2)]
    else:
       df.loc[i] = [round(i), 'b', round(item['numInp']), round(sm.cnum), round(sm.bias,2), sm.svrc, sm.svrp, round(sm.trnq2,2), round(sm.chkq2,2)]
    df.to_csv('result.csv', encoding='utf-8', index=False)
    i += 1

print("ok")
selfea
np.shape(selfea)
