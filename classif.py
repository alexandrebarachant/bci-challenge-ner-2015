# -*- coding: utf-8 -*-
"""
@author: alexandrebarachant
"""
import numpy
from scipy.linalg import eig as geig
import riemann
from sklearn.base  import BaseEstimator, ClassifierMixin, TransformerMixin
###############################################################################
class XdawnCovariances(BaseEstimator,TransformerMixin):
    """ 
    Compute double xdawn, project the signal and compute the covariances

    """    
    def __init__(self,nfilter=4,subelec=-1):
        self.nfilter = nfilter
        self.subelec = subelec
        
    def fit(self,X,y):
        Nt,Ne,Ns = X.shape
        # Prototyped responce for each class
        P1 = numpy.mean(X[y==1,:,:],axis=0)
        P0 = numpy.mean(X[y==0,:,:],axis=0)
        
        # Covariance matrix of the prototyper response & signal
        C1 = numpy.matrix(numpy.cov(P1))
        C0 = numpy.matrix(numpy.cov(P0))
        
        #FIXME : too many reshape operation        
        tmp = X.transpose((1,2,0))
        Cx = numpy.matrix(numpy.cov(tmp.reshape(Ne,Ns*Nt)))
        
        # Spatial filters
        D,V1 = geig(C1,Cx)        
        D,V0 = geig(C0,Cx)
        
        # create the reduced prototyped response
        self.P = numpy.concatenate((numpy.dot(V1[:,0:self.nfilter].T,P1),numpy.dot(V0[:,0:self.nfilter].T,P0)),axis=0)
    
    def transform(self,X):
        covmats = riemann.covariances_EP(X[:,self.subelec,:],self.P)
        return covmats
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
###############################################################################
class TangentSpace(BaseEstimator, TransformerMixin):

    def __init__(self,metric='riemann',tsupdate = False):

        self.metric = metric
        self.tsupdate = tsupdate 

        
    def fit(self,X,y=None):
        # compute mean covariance
        self.Cr = riemann.mean_covariance(X,metric=self.metric)
        
    def transform(self,X):
       
        if self.tsupdate:
            Cr = riemann.mean_covariance(X,metric=self.metric)
        else:
            Cr = self.Cr
        return riemann.tangent_space(X,Cr)

    def fit_transform(self,X,y=None):
        # compute mean covariance
        self.Cr = riemann.mean_covariance(X,metric=self.metric)
        return riemann.tangent_space(X,self.Cr)
###############################################################################
class AddMeta(BaseEstimator, TransformerMixin):

    def __init__(self,meta=None):
        self.meta = meta
    
    def fit(self,X,y=None):
        pass
        
    def transform(self,X):
        if self.meta is not None:
            return numpy.c_[X,self.meta]
        else:
            return X

    def fit_transform(self,X,y=None):
        return self.transform(X)

###############################################################################
class ElectrodeSelect(BaseEstimator, TransformerMixin):

    def __init__(self,nelec = 20,nfilters=5,metric='riemann'):
        self.nelec = nelec
        self.metric = metric
        self.nfilters = nfilters
        self.subelec = -1
        self.dist = []
    
    def fit(self,X,y=None):
        C1 = riemann.mean_covariance(X[y==1,...],self.metric)
        C0 = riemann.mean_covariance(X[y==0,...],self.metric)
        
        Ne,_ = C0.shape
        
        self.subelec = range(0,Ne,1) 
        while (len(self.subelec)-2*self.nfilters)>self.nelec:
            di = numpy.zeros((len(self.subelec),1))
            for idx in range(2*self.nfilters,len(self.subelec)):
                sub = self.subelec[:]
                sub.pop(idx)
                di[idx] = riemann.distance(C0[:,sub][sub,:],C1[:,sub][sub,:])
            #print di
            torm = di.argmax()
            self.dist.append(di.max())
            self.subelec.pop(torm)        
        #print self.subelec
        
    def transform(self,X):
       return X[:,self.subelec,:][:,:,self.subelec]

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)
###############################################################################
def updateMeta(clf,Meta):
    if clf.named_steps.has_key('addmeta'):
        clf.set_params(addmeta__meta=Meta) 

def baggingIterator(opts,users):
    mdls = opts['bagging']['models']
    bag_size = 1-opts['bagging']['bag_size']
    bag_size = numpy.floor(bag_size*len(users))
    if bag_size == 0:
        return [[u] for u in users]
    else:
        return [numpy.random.choice(users,size=bag_size,replace=False) for i in range(mdls)]