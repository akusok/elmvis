#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double getdiff(double *A, double *Y, int d, int N, int i1, int i2, int nthr);
