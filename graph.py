# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:16:03 2015

@author: akusok
"""

import numpy as np
from matplotlib import pyplot as plt

#data = np.vstack((data, np.array([d, tc, tp])))
data = np.loadtxt("mydata.txt")
d = data[:, 0]
tc = data[:, 1]
tp = data[:, 2]

plt.plot(d, tc, '--b')
plt.plot(d, tp, '-r')
plt.ylabel("Iters per second")
plt.yscale('log')
plt.xlabel("L")
plt.show()
    
