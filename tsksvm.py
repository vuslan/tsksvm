# tsksvm is the package of this project

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