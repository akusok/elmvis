# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
import hpelm
import cPickle
from time import time
#from elmvis_cython import elmvis
#from elmvis_python import elmvis
#from elmvis_gpu import elmvis as elmvis_gpu
from elmvis_hybrid import elmvis32 as elmvis

#from elmvis_ga import GA


maxiter = 2001
stall = 1000000
report = 1000

if True:
    X = cPickle.load(open("artiface_orig.pkl", "rb"))
    np.random.shuffle(X)
    X = X[:, X.std(0) > 1E-6]
    X = (X - X.mean(0)) / X.std(0)
    X = np.ascontiguousarray(X)
else:
    X = np.genfromtxt("DATA.txt", delimiter=',')
    X = (X - X.mean(0)) / np.linalg.norm(X, axis=0)

#d = 3
#X = X[:, :d]

#X = np.random.rand(5000, 4000)

N, d = X.shape
V = np.random.rand(N, 2)*2 - 1
V = (V - V.mean(0)) / V.std(0)

# build initial ELM
L = int(d**0.5)+1
elm = hpelm.ELM(2, d)
elm.add_neurons(2, 'lin')
elm.add_neurons(L, 'sigm')
#elm.train(V, X)
#print "L=%d:  %.5f" % (L, elm.error(elm.predict(V), X))

# compute initial A
H = elm.project(V)
A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

tol = 1E-6
t = time()
X2, cost, iters, updates = elmvis(X.astype(np.float32),
                                  A.astype(np.float32),
                                  tol=tol,
                                  maxiter=maxiter,
                                  maxstall=stall,
                                  maxupdate=10000,
                                  report=report)
t = time()-t
print iters, updates

elm.train(V, X)
MSE0 = elm.error(elm.predict(V), X)

#for _ in xrange(2):
#    # initialize X with GA
#    p, sim = GA(X, A, pop=200, gen=30, verbose=False)
#    X = X[p]
#    elm.train(V, X)
#    MSE1 = elm.error(elm.predict(V), X)
#
#    # fine-tune X
#    X, cost = opt(X, A, iter, stall, report)
#    elm.train(V, X)
#    MSE2 = elm.error(elm.predict(V), X)
#
print "%.5f" % MSE0
#print MSE1
#print MSE2
print 'Done!'
















