
# coding: utf-8

# In[1]:


def findq2(expout,prdout):

   nSamples = expout.size;

   sum1 = 0
   for i in range(1,nSamples):
       c = expout[i]-prdout[i]
       d = np.square(c)
       sum1 = sum1 + d

   sum2 = 0
   e = np.mean(expout)
   for i in range(1,nSamples):
       f = expout[i] - e
       g = np.square(f)
       sum2 = sum2 + g

   q2 = 1-(sum1/sum2)

   return q2


# In[2]:


def gaussmf(x,sigma,c):
    dividend = -1*np.square(x - c)
    divisor = 2*np.square(sigma)
    quotient = dividend/divisor;
    y=np.exp(quotient)
    return y


# In[3]:


def trainfsvm(X,y,M,sigma,ker,svrc,svrp):
   # L is the number of data samples
   L=X.shape[0] #gives number of row count
   # m is the number of rules
   m=M.shape[0]
   # n is the number of features
   n=X.shape[1] #gives number of col count

   out = [] # from copy

   trn_labels = y;
   trn_features = np.zeros((L,m*(n+1)))

   trnA = np.zeros((L,m*(n+1)))
   weights = np.zeros((L,m))

   itermax = 1

   for iter in range(0,itermax):
       for i in range(0,L):
           U=[]
           for j in range(0,m):
               u=1
               for t in range(0,n):
                   u=u*(gaussmf(X[t][i],sigma[t][j],M[t][j]))
               U=U+[u]
           fa=U/sum(U) # this is the weight
           row = np.append(X.iloc[[i]].values,1)
           xtemp = np.zeros(shape=(fa.size,row.size))
           for ii in range(0,fa.size):
               for jj in range(0,row.size):
                   xtemp[ii][jj]= fa[ii]*row[jj]
           xtemp=np.reshape(xtemp, fa.size*row.size)
           trnA.transpose()[:,i] = xtemp;
           weights.transpose()[:,i] = fa;
       trn_features = trnA; # X #trn_labels = y
       clf = svr(kernel=ker, C=svrc, epsilon=svrp)
       clf.fit(trn_features, trn_labels)
       w=np.dot(clf.support_vectors_.transpose(),clf.dual_coef_.transpose())
       bias = clf.intercept_
       C = np.reshape(w,(m,n+1))
       C = pd.DataFrame(C)
   return clf, C, bias


# In[4]:


def predictfsvm(X,y,M,sigma,C,bias):
   # L is the number of data samples
   L=X.shape[0] #gives number of row count
   # m is the number of rules
   m=M.shape[0]
   # n is the number of features
   n=X.shape[1] #gives number of col count

   out = []
   labels = y;

   for i in range(0,L):
       U=[]
       #print "i="+str(i)
       for j in range(0,m):
           u=1
           for t in range(0,n):
               u=u*(gaussmf(X[t][i],sigma[t][j],M[t][j]))
           U=U+[u];
           #print "j="+str(j)+" "+"U="+str(U)
       fa=U/sum(U); # this is the weight
       c0=C[n]
       for t in range(0,n):
           c0=c0+C[t]*X[t][i] # calculating y for each rule
       c0=c0.as_matrix()
       f=np.dot(fa,c0) + bias
       out = out + [f]
   return out


# In[5]:


# The inital set of amino acids and their numerical values
def aa2int(aa):
    aalist = "ARNDCQEGHILKMFPSTWYVBZX*-"
    aalist = list(aalist)
    return aalist.index(aa)


# In[6]:


def pep2mat(peptides):
    list_of_peptides = list(peptides)
    num_of_peptides = len(list_of_peptides)
    pepsize = len(list_of_peptides[0])
    matrix = np.ones((num_of_peptides,pepsize))
    for i in range(num_of_peptides):
        peptide = list(list_of_peptides[i])
        for j in range(pepsize):
            matrix[i,j] = aa2int(peptide[j])
    return matrix


# In[7]:


def peptide2scales(numPeptides,nFeatures):
    nSamples = numPeptides.shape[0]
    nAA = numPeptides.shape[1]
    datin = -1*np.ones((nSamples,nFeatures*nAA))
    for i in range(nSamples):
        for j in range(nAA):
            aa = int(numPeptides[i,j])
            for k in range(nFeatures):
                idx = j*nFeatures + k
                datin[i,idx] = scales[k,aa]
    return pd.DataFrame(datin)


# In[8]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR as svr
from sklearn.preprocessing import MinMaxScaler as mmscaler
from sklearn.feature_selection import VarianceThreshold as vt
import skfuzzy as fuzz


# ###### LOAD AA SCALES

# In[9]:


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

numInp=len(train.shape[1])

Xtrain = trnin.as_matrix()
Xtrain = pd.DataFrame(Xtrain)
Xtest = chkin.as_matrix()
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

