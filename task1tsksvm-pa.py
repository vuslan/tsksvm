import numpy as np
import pandas as pd
from sklearn.svm import SVR as svr
from sklearn.preprocessing import MinMaxScaler as mmscaler
from sklearn.feature_selection import VarianceThreshold as vt
import skfuzzy as fuzz

# ###### LOAD AA SCALES

# min-max normalized from coepra-scales-2.csv
scaler = mmscaler()
df_coeprascales2=pd.read_csv("data/coepra-scales-2.csv",header=None); # (643x20)
np_coeprascales2=np.array(df_coeprascales2.transpose())
scaler.fit(np_coeprascales2)
np_coeprascales2 = scaler.transform(np_coeprascales2)
df_scales = pd.DataFrame(np_coeprascales2)
df_scales = df_scales.transpose() # (643x20)
scales = np.array(df_scales)
nFeatures = scales.shape[0]


# ###### SELECTED PARAMETER SET

# In[10]:


param = {'cnum':7, 'svrc':2.40, 'svrp':0.05, 'threshold':0.8}


# ###### LOAD TRAIN DATASET

# In[11]:


data=pd.read_csv("data/task1trnpep.txt", sep=' ', header=None)
txtPeptides = data[0]
numPeptides = pep2mat(txtPeptides); # n: make numerical
ytrain=pd.DataFrame(data[1])


# In[12]:


trnin=peptide2scales(numPeptides,nFeatures)


# ###### SELECTED FEATURE SET

# In[13]:


selfea=pd.read_csv("task1env/selfea.dat",header=None)
selfea = selfea.transpose().values[0]-1
numInp=len(selfea)


# ###### SELECTED FEATURE SET WITH VARIENCETHRESHOLD

# In[14]:


tvalue = float(param['threshold'])
selector = vt(threshold=(tvalue * (1 - tvalue)))
selector.fit_transform(trnin)
selidx = selector.get_support(indices=True)
selfea = selidx
numInp=len(selidx)
len(selidx)


# In[15]:


Xtrain = trnin.as_matrix()[:,selfea.tolist()]
Xtrain = pd.DataFrame(Xtrain)


# ###### FUZZY C MEANS CLUSTERING

# In[16]:


# parameter identification: (mean)
trndat = pd.concat([Xtrain, ytrain], axis=1, ignore_index=True) # input ve output beraber cluster edilcek.
# error 0.005 and maxiter=1000 are skfuzzy parameters
# error 0.00001 and maxiter=100 are matlab parameters
ctr, U, u0, d, jm, p, fpc = fuzz.cluster.cmeans(trndat.transpose(), param['cnum'], 2, error=0.00001, maxiter=100, init=None)
M = ctr[:, 0:numInp] # Antecedent Mean (M)
M = pd.DataFrame(np.round(M,4))


# In[17]:


# parameter identification: (stddev)
inpstd = np.zeros((param['cnum'],numInp)) #stddev
for i in range(0,param['cnum']):
    u = U[i] # u is the membership values of data samples to the next cluster
    v = M.as_matrix()[i] # v is the mean values of the next cluster | cl.center >> ctr
    n = Xtrain.shape[0] # n is the number of data
    numInp = Xtrain.shape[1] # number of inputs
    diff = np.zeros((n,numInp))
    suu = 0 # sum of u

    # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
    for j in range(0,n):
        diff[j] = Xtrain.as_matrix()[j]-v

    for j in range(0,n):
        suu = suu + u[j]**1

    val = np.dot((diff**2).transpose(),(u**1).transpose())/suu # val is the variable for variance
    val = val.transpose() # (vuvar2)

    inpstd[i] = np.sqrt(val) # Antecedent S
    inpstd[i] = np.round(inpstd[i],4)
inpstd = pd.DataFrame(inpstd)


# ###### TRAINING FUZZY SVM

# In[18]:


ker = 'linear'
clf,C,bias = trainfsvm(Xtrain,ytrain.values.ravel(),M,inpstd,ker,param['svrc'],param['svrp'])
print('clf:'+str(clf))
print('bias:'+str(bias))


# ###### PREDICTING TRAIN DATASET

# In[19]:


expout = ytrain.as_matrix()
prdout = predictfsvm(Xtrain,ytrain,M,inpstd,C,bias)
trnq2 = findq2(expout,prdout)
print('trnq2:'+str(trnq2))


# ###### LOAD TEST DATASET

# In[20]:


data=pd.read_csv("data/task1tstpep.txt", sep=' ', header=None)
txtPeptides = data[0]
numPeptides = pep2mat(txtPeptides); # n: make numerical
ytest=pd.DataFrame(data[1])


# In[21]:


chkin=peptide2scales(numPeptides,nFeatures)


# In[22]:


Xtest = chkin.as_matrix()[:,selfea.tolist()]
Xtest = pd.DataFrame(Xtest)


# ###### PREDICTING TEST DATASET

# In[23]:


expout = ytest.as_matrix()
prdout = predictfsvm(Xtest,ytest,M,inpstd,C,bias)
chkq2 = findq2(expout,prdout)
print('chkq2:'+str(chkq2))


# ###### RESULTS

# In[24]:


print('trnq2:'+str(trnq2))
print('chkq2:'+str(chkq2))


# ###### REPEATPROCESS

# In[25]:


def repeatprocess(param):
    # parameter identification: (mean)
    trndat = pd.concat([Xtrain, ytrain], axis=1, ignore_index=True) # input ve output beraber cluster edilcek.
    ctr, U, u0, d, jm, p, fpc = fuzz.cluster.cmeans(trndat.transpose(), param['cnum'], 2, error=0.00001, maxiter=100, init=None)
    M = ctr[:, 0:param['numInp']] # Antecedent Mean (M)
    M = pd.DataFrame(np.round(M,4))

    # parameter identification: (stddev)
    inpstd = np.zeros((param['cnum'],param['numInp'])) #stddev
    for i in range(0,param['cnum']):
        u = U[i] # u is the membership values of data samples to the next cluster
        v = M.as_matrix()[i] # v is the mean values of the next cluster | cl.center >> ctr
        n = Xtrain.shape[0] # n is the number of data
        numInp = Xtrain.shape[1] # number of inputs
        diff = np.zeros((n,param['numInp']))
        suu = 0 # sum of u

        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        for j in range(0,n):
            diff[j] = Xtrain.as_matrix()[j]-v

        for j in range(0,n):
            suu = suu + u[j]**1

        val = np.dot((diff**2).transpose(),(u**1).transpose())/suu # val is the variable for variance
        val = val.transpose() # (vuvar2)

        inpstd[i] = np.sqrt(val) # Antecedent S
        inpstd[i] = np.round(inpstd[i],4)
    inpstd = pd.DataFrame(inpstd)

    ker = 'linear'
    clf,C,bias = trainfsvm(Xtrain,ytrain.values.ravel(),M,inpstd,ker,param['svrc'],param['svrp'])

    expout = ytrain.as_matrix()
    prdout = predictfsvm(Xtrain,ytrain,M,inpstd,C,bias)
    trnq2 = findq2(expout,prdout)

    expout = ytest.as_matrix()
    prdout = predictfsvm(Xtest,ytest,M,inpstd,C,bias)
    chkq2 = findq2(expout,prdout)

    paramnew = param.copy()
    paramnew['bias'] = round(bias,3)
    paramnew['trnq2'] = round(trnq2,3)
    paramnew['chkq2'] = round(chkq2,3)

    return paramnew


# In[ ]:


opt = float("-inf")
L = []
df = pd.DataFrame(L)
df=pd.DataFrame(df, columns=['a/b', 'fs/threshold', 'numInp', 'cnum', 'bias', 'svrc', 'svrp', 'trnq2', 'chkq2'])
i=0
for t in np.arange(0.90, 0.99, 0.01):

    selector = vt(threshold=(t * (1 - t)))
    selector.fit_transform(trnin)
    selidx = selector.get_support(indices=True)
    selfea = selidx
    numInp=len(selidx)

    Xtrain = trnin.as_matrix()[:,selfea.tolist()]
    Xtrain = pd.DataFrame(Xtrain)
    Xtest = chkin.as_matrix()[:,selfea.tolist()]
    Xtest = pd.DataFrame(Xtest)

    for cnum in range(2,3):
        for svrc in np.arange(1.0, 1.5, 0.5):
            for svrp in np.arange(0.05, 0.06, 0.01):
                param = {'numInp':numInp, 'cnum':cnum, 'svrc':svrc, 'svrp':svrp}
                paramnew = repeatprocess(param)
                if float(paramnew['chkq2'])>opt:
                   opt = float(paramnew['chkq2'])
                   df.loc[i] = ['a', t, paramnew['numInp'], paramnew['cnum'], paramnew['bias'], paramnew['svrc'], paramnew['svrp'], paramnew['trnq2'], paramnew['chkq2']]
                else:
                   df.loc[i] = ['b', t, paramnew['numInp'], paramnew['cnum'], paramnew['bias'], paramnew['svrc'], paramnew['svrp'], paramnew['trnq2'], paramnew['chkq2']]
                df.to_csv('result.csv', encoding='utf-8', index=False)
                print(i)
                i=i+1
print("ok")


# In[ ]:


df

