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
from elmvis_cython import elmvis_cython
from elmvis_python import elmvis
from elmvis_gpu import elmvis_gpu
from elmvis_hybrid import elmvis_hybrid
#from elmvis_ga import GA


def run():
    data = np.empty((0, 9))
    d = int(sys.argv[2])
    k2 = 1000
    
    for N in range(20, 10000+1, 20):
        print 'processing N=%d (k=%d)' % (N, k2),
        X = np.random.rand(N, d)
        
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
        
        # find reasonable problem size
#        k = 2
#        while True:
#            k *= 2
#            t = time()
#            elmvis_gpu(X, A, tol=1E+9, maxiter=k, maxstall=k, report=k)            
#            elmvis(X, A, tol=1E+9, maxiter=k, maxstall=k, report=k)  # always check
#            t = time()-t
#            if t > 1:
#                break
        
#        print "time python: %f, report every %d" % (t, k)
        k10 = k2*10
        
        # test all methods
        _,_,p1 = elmvis(X, A, tol=1E+9, maxiter=k2, maxstall=k2, report=k2/2)
        _,_,p2 = elmvis(X, A, tol=-1E+9, maxiter=k10, maxstall=k10, report=k10/2)
    
        _,_,c1 = elmvis_cython(X, A, tol=1E+9, maxiter=k2, maxstall=k2, report=k2/2)
        _,_,c2 = elmvis_cython(X, A, tol=-1E+9, maxiter=k10, maxstall=k10, report=k10/2)
    
        _,_,g1 = elmvis_gpu(X, A, tol=1E+9, maxiter=k2, maxstall=k2, report=k2/2)
        _,_,g2 = elmvis_gpu(X, A, tol=-1E+9, maxiter=k10, maxstall=k10, report=k10/2)
    
        _,_,h1 = elmvis_hybrid(X, A, tol=1E+9, maxiter=k2, maxstall=k2, report=k2/2)
        _,_,h2 = elmvis_hybrid(X, A, tol=-1E+9, maxiter=k10, maxstall=k10, report=k10/2)
    
        data = np.vstack((data, np.array([N, p1, p2, c1, c2, g1, g2, h1, h2])))
        np.savetxt(sys.argv[1], data, fmt="%.3f")
        k2 = int(min([p1, p2, c1, c2, g1, g2, h1, h2]))
    
run()















