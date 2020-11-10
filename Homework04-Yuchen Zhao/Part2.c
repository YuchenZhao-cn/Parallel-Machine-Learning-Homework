//{
//omp_id = omp_get_thread_num();
//if (omp_id == 0)
//{
//nthreads = omp_get_num_threads();
//}
// 
// 
//  // parallelize the for loop using dynamic schedule
//for (i=0; i<N; i++)
//{
//tmp = 2.0* a[i];
//a[i] = tmp;
//c[i] = a[i] + b[i];
//}
//
//}  

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100

int main (int argc, char *argv[]) {

int nthreads, omp_id, i, tmp;
float a[N], b[N], c[N];

/* Some initializations */
for (i = 0; i < N; i++)
a[i] = b[i] = i;

#pragma omp parallel shared(a,b,c,nthreads) private(i, omp_id, tmp)
{
omp_id = omp_get_thread_num();
if(omp_id == 0)
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}

printf("Thread %d starting...\n", omp_id);

#pragma omp parallel for schedule(dynamic)
for(i = 0; i < N; i++)
{
tmp = 2.0* a[i];
a[i] = tmp;
c[i] = a[i] + b[i];
printf("Thread %d: c[%d] = %f\n", omp_id, i, c[i]);
}

} /* end of parallel section */

}
