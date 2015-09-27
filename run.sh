#!/bin/bash

# rm *.so
# rm elmvis_opt.c
python elmvis_opt_setup.py build_ext --inplace
python elmvisplus.py 1
python elmvisplus.py 2
python elmvisplus.py 3
python elmvisplus.py 4
