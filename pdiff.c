#include <stdbool.h>
#include <omp.h>

long pdiff(double *A, double *X, double *AX, long *ari1, long *ari2, double tol, double *diff, long *iters, long d, long N, long rep)
{
    long r, j, i1, i2, iopt=-1, myiters=0;
    double xi1, xi2, diff1;
    bool done = false;
    //omp_set_num_threads(2);

    #pragma omp parallel for shared(X, A, d, N, iopt) private(r, j, i1, i2, xi1, xi2, diff1) reduction(|:done) reduction(+:myiters)
    for (r = 0; r < rep; r++)
    {
        i1 = ari1[r];
        i2 = ari2[r];
        diff1 = 0;
        for (j = 0; j < d; j++)
        {
            xi1 = X[i1*d + j];
            xi2 = X[i2*d + j];
            diff1 += (A[i1*N + i1] * (xi2 - xi1) + 2*AX[i1*d + j]) * (xi2 - xi1) +
                     (A[i2*N + i2] * (xi1 - xi2) + 2*(AX[i2*d + j] + A[i2*N + i1]*(xi2 - xi1))) * (xi1 - xi2);
        }
        myiters += 1;
        if (diff1 > tol)
        {
            done = true;
            #pragma omp critical
            {
                *diff = diff1;
                iopt = r;
            }
        }
        if (done) r = rep;
    }
    *iters = myiters;
    return iopt;
}
