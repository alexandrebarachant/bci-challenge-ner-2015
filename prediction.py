# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 22:23:02 2014

@author: kirsh
"""

from time import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
import yaml
import sys
from classif import updateMeta, baggingIterator
from multiprocessing import Pool
from functools import partial

def from_yaml_to_func(method,params):
    prm = dict()
    if params!=None:
        for key,val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


def BaggingFunc(bag,Labels,X,Meta,User,X_test,Meta_test,User_test):
    bagUsers = np.array([True if u in set(bag) else False for u in User])
    train_index = np.negative(bagUsers)
    updateMeta(clf,Meta[train_index])
    clf.fit(X[train_index,:,:],Labels[train_index])
    
    ### predicting
    prob = []
    for ut in users_test:
        updateMeta(clf,Meta_test[User_test==ut,...])
        prob.extend(clf.predict(X_test[User_test==ut,...]))
    prob = np.array(prob)
    
    return prob

# load parameters file
yml = yaml.load(open(sys.argv[1]))

# imports 
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['pipeline']:
    for method,params in item.iteritems():
        pipe.append(from_yaml_to_func(method,params))

# create pipeline
clf = make_pipeline(*pipe)
opts=yml['MetaPipeline']
if opts is None:
    opts = {}

cores = yml['Submission']['cores']

# load data
X = np.load('./preproc/epochs.npy')
Labels,User = np.load('./preproc/infos.npy')
users = np.unique(User)
Meta = np.load('./preproc/meta_leak.npy') if opts.has_key('leak') else np.load('./preproc/meta.npy')

X_test = np.load('./preproc/test_epochs.npy')
feedbackid,User_test = np.load('./preproc/test_infos.npy')
User_test = np.array(map(int, User_test))
users_test = np.unique(User_test)
Meta_test = np.load('./preproc/test_meta_leak.npy') if opts.has_key('leak') else np.load('./preproc/test_meta.npy')

### training
np.random.seed(5)
allProb = 0 

if opts.has_key('bagging'):
    bagging = baggingIterator(opts,users)
else:
    bagging = [[-1]]

t = time()
pBaggingFunc = partial(BaggingFunc,Labels=Labels,X=X,Meta=Meta,User=User,X_test=X_test,Meta_test=Meta_test,User_test=User_test)
pool = Pool(processes = cores)
allProb = pool.map(pBaggingFunc,bagging,chunksize=1)
allProb = np.vstack(allProb)
allProb = np.mean(allProb,axis=0)

if opts.has_key('leak'):
    allProb += opts['leak']['coeff']*(1-Meta_test[:,-1])

print "Done in " + str(time()-t) + " second"

submission = yml['Submission']['path']
df = pd.DataFrame({'IdFeedBack':feedbackid,'Prediction':allProb})
df.to_csv(submission,index=False)