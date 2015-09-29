# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:20:36 2015

@author: akusok
"""

import numpy as np
import hpelm
import cPickle
import sys
from time import time
from multiprocessing import cpu_count
from elmvisopt import opt as c_opt, bench as c_bench
from elmvis_python import opt as p_opt, bench as p_bench
#from elmvis_ga import GA


maxiter = 10000000
stall = 10000
report = 10000


data = np.empty((0,5))

for d in range(2, 1000):
    print 'processing d=%d' % d
    X = np.random.rand(1000, d)
    
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
    
    # find best method
    #ncores = cpu_count()
    k = 2
    while True:
        k *= 2
        tc,_ = c_bench(X, A, maxiter=k, nthreads=1)
        tp,_ = p_bench(X, A, maxiter=k)
        if max((tc, tp)) > 1:
            break
    
    opt = c_opt if tc < tp else p_opt
    report = int(k*1.0/min((tc, tp)))
    report = k
    print "time C: %f, time python: %f, report every %d" % (tc, tp, report)
    
    # iteratively tune X
    maxiter = report*3
    stall = report*100

    _,_,c1 = c_opt(X, A, -1E+9, maxiter, stall, report=report, nthreads=1)

    _,_,c2 = c_opt(X, A, -1E+9, maxiter, stall, report=report, nthreads=2)

    _,_,c4 = c_opt(X, A, -1E+9, maxiter, stall, report=report, nthreads=4)

    _,_,p4 = p_opt(X, A, -1E+9, maxiter, stall, report=report)

    data = np.vstack((data, np.array([d, c1, c2, c4, p4])))
    np.savetxt(sys.argv[1], data, fmt="%.3f")















