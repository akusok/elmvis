# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:52:14 2015

@author: akusok
"""

import numpy as np
import hpelm
from elmvis_ga import GA
from elmvisopt import opt
#from elmvis_python import opt

iter = 10000000
stall = 10000
report = 100000


X = np.genfromtxt("DATA.txt", delimiter=',')
N, d = X.shape
X = (X - X.mean(0)) / np.linalg.norm(X, axis=0)
V = np.random.rand(N, 2)*2 - 1
V = (V - V.mean(0)) / V.std(0)

# build initial ELM
elm = hpelm.ELM(2, d)
elm.add_neurons(2, 'lin')
elm.add_neurons(28, 'sigm')
elm.train(V, X)
print elm.error(elm.predict(V), X)

# compute initial A
H = elm.project(V)
A = H.dot(np.linalg.pinv(np.dot(H.T, H))).dot(H.T)

# iteratively tune X
X, cost = opt(X, A, iter, stall, report)
elm.train(V, X)
MSE0 = elm.error(elm.predict(V), X)

for _ in xrange(2):
    # initialize X with GA
    p, sim = GA(X, A, pop=200, gen=30, verbose=False)
    X = X[p]
    elm.train(V, X)
    MSE1 = elm.error(elm.predict(V), X)

    # fine-tune X
    X, cost = opt(X, A, iter, stall, report)
    elm.train(V, X)
    MSE2 = elm.error(elm.predict(V), X)

print MSE0
print MSE1
print MSE2
print 'Done!'
















