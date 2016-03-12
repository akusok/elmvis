# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:18:04 2015

@author: akusok
"""

import numpy as np
import hashlib

class FastIX():
    def __init__(self, X):
        self.Xh = {hashlib.sha1(X[k]).hexdigest(): k for k in range(X.shape[0])}

    def ix(self, X2):
        return np.array([self.Xh[hashlib.sha1(x).hexdigest()] for x in X2])
