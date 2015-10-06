# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:21:14 2015

@author: akusok
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("elmvis_cython",
                         ["elmvis_cython.pyx"],
                         include_dirs=[numpy.get_include()])]

setup(
  name = 'ELMVIS+ cython app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

print
print "Done!"