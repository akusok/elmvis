# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
import hpelm
import cPickle
import sys, os
from elmvisopt import opt
from multiprocessing import Process, Queue
#from elmvis_ga import GA
#from elmvis_python import opt


# run as:
# OMP_NUM_THREADS=1 python elmvisplus.py 4




maxiter = 10000000
stall = 1000
report = 1000

if True:
    X = cPickle.load(open("artiface_orig.pkl", "rb"))
    np.random.shuffle(X)
    X = X[:, X.std(0) > 1E-6]
    X = (X - X.mean(0)) / X.std(0)
else:
    X = np.genfromtxt("DATA.txt", delimiter=',')
    X = (X - X.mean(0)) / np.linalg.norm(X, axis=0)

N, d = X.shape
V = np.random.rand(N, 2)*2 - 1
V = (V - V.mean(0)) / V.std(0)

# build initial ELM
L = 950
elm = hpelm.ELM(2, d)
elm.add_neurons(2, 'lin')
elm.add_neurons(L, 'sigm')
elm.train(V, X)
print "L=%d:  %.5f" % (L, elm.error(elm.predict(V), X))

# compute initial A
H = elm.project(V)
A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

# iteratively tune X
X, cost = opt(X, A, 1E-9, maxiter, stall, report, nthreads=int(sys.argv[1]))
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
















