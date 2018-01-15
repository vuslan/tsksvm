# tsksvm is the package of this project

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
import skfuzzy


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


def gaussmf(x,sigma,c):
    dividend = -1*np.square(x - c)
    divisor = 2*np.square(sigma)
    quotient = dividend/divisor;
    y=np.exp(quotient)
    return y


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
       clf = SVR(kernel=ker, C=svrc, epsilon=svrp)
       clf.fit(trn_features, trn_labels)
       w=np.dot(clf.support_vectors_.transpose(),clf.dual_coef_.transpose())
       bias = clf.intercept_
       C = np.reshape(w,(m,n+1))
       C = pd.DataFrame(C)
   return clf, C, bias


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


# The inital set of amino acids and their numerical values
def aa2int(aa):
    aalist = "ARNDCQEGHILKMFPSTWYVBZX*-"
    aalist = list(aalist)
    return aalist.index(aa)


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


def peptide2scales(numPeptides,scales):
    nSamples = numPeptides.shape[0]
    nAA = numPeptides.shape[1]
    nFeatures = scales.shape[0]
    datin = -1*np.ones((nSamples,nFeatures*nAA))
    for i in range(nSamples):
        for j in range(nAA):
            aa = int(numPeptides[i,j])
            for k in range(nFeatures):
                idx = j*nFeatures + k
                datin[i,idx] = scales[k,aa]
    return pd.DataFrame(datin)


# input: scales_file (as num_of_index x num_of_aa)
# example file: coepra-scales-2.csv
# output: np_scales (numpy array of scales)
def preprocess_aa_scales(scales_file):
    # read the scales from the csv file as a dataframe
    # rows: num of scales
    # cols: num of aa (=20)
    df_scales = pd.read_csv(scales_file, header=None)
    # . transpose dataframe so that
    # samples should be aa and features should be aaindex
    # rows: num of aa (=20)
    # cols: num of scales
    tp_df_scales = df_scales.transpose()
    # convert dataframe to numpy array
    np_scales = np.array(tp_df_scales)
    # . this is minmax scaler object that transforms features
    # by scaling each feature to a given range.
    # . the default range of tranformed data is (0, 1).
    # . http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    scaler = MinMaxScaler()
    # compute the minimum and maximum aaindex values to be used for later scaling.
    scaler.fit(np_scales)
    # convert aaindex values to minmax values
    tf_np_scales = scaler.transform(np_scales)
    # convert to the input file format (as num_of_index x num_of_aa)
    tp_tf_np_scales = tf_np_scales.transpose()
    return tp_tf_np_scales


def load_dataset(data_file):
    data=pd.read_csv(data_file, sep=' ', header=None)
    txtPeptides = data[0]
    numPeptides = pep2mat(txtPeptides); # n: make numerical
    y = pd.DataFrame(data[1])
    return numPeptides, y

def mcfs(trnin, num_fea):

    from skfeature.utility import construct_W
    kwargs_W = {"metric":"euclidean", "neighbor_mode":"knn", "weight_mode":"heat_kernel", "k":5, 't':1}
    W = construct_W.construct_W(trnin, **kwargs_W)

    from skfeature.function.sparse_learning_based import MCFS
    score = MCFS.mcfs(trnin, num_fea, W=W)
    idx = MCFS.feature_ranking(score)
    selfea = idx[0:num_fea]
    return selfea


# https://www.madewithtea.com/automatic-parameter-tuning-for-machine-learning.html
# https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
# https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
# https://stats.stackexchange.com/questions/91842/how-to-perform-parameters-tuning-for-machine-learning
# http://scikit-learn.org/stable/modules/grid_search.html
# https://wiki.python.org/moin/Iterator
# https://www.tutorialspoint.com/python/python_tuples.htm
class SelectModel:
    cnum = 2
    svrc = 1.0
    svrp = 0.05
    numInp = 0
    bias = 0.0
    trnq2 = 0.0
    chkq2 = 0.0
    # paramnew['bias'] = float(np.round(bias,decimals=3))
    def __init__(self, cnum, svrc, svrp):
        self.cnum = cnum
        self.svrc = svrc
        self.svrp = svrp
    def select_features_with_variencethreshold(self, trnin, threshold):
        tvalue = threshold
        # pdb.set_trace()
        selector = VarianceThreshold(threshold=(tvalue * (1 - tvalue)))
        selector.fit_transform(trnin)
        selidx = selector.get_support(indices=True)
        selfea = selidx
        numInp = len(selidx)
        return selfea, numInp


    def identify_fuzzy_system_with_fcm(self, Xtrain, ytrain, cnum):
        self.cnum = cnum
        trndat = pd.concat([Xtrain, ytrain], axis=1, ignore_index=True) # input ve output beraber cluster edilcek.
        # http://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html
        # error 0.005 and maxiter=1000 are skfuzzy parameters
        # error 0.00001 and maxiter=100 are matlab parameters
        ctr, U, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(trndat.transpose(), self.cnum, 2, error=0.005, maxiter=1000, init=None)
        # parameter identification: (mean)
        M = ctr[:, 0:Xtrain.shape[1]] # Antecedent Mean (M)
        M = pd.DataFrame(np.round(M,4))
        # parameter identification: (stddev)
        inpstd = np.zeros((self.cnum,Xtrain.shape[1])) #stddev
        for i in range(0,self.cnum):
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
        return M, inpstd
