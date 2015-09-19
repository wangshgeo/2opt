#ifndef __OPT2GPU_H__
#define __OPT2GPU_H__

#include "include/instance.h"

#ifdef __cplusplus
extern "C" {
#endif

void initCudaArray (dtype **d_A, dtype *h_A, unsigned int N);
void cudaOpt2 (COORD *h_coords,COORD *d_coords, unsigned int nc);
void reduce (dtype* A, unsigned int N, unsigned int OPT, dtype* ret);
void initCudaTour (int **d_A, int *h_A, unsigned int N);
void initCudaCoords (COORD **d_A, COORD*h_A, unsigned int N);

#ifdef __cplusplus
}
#endif

#endif
