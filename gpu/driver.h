#ifndef __DRIVER_H__
#define __DRIVER_H__

typedef float dtype;
#include "include/instance.h"

dtype reduceCpu(dtype*h_A,unsigned int N);
void swapCoords(COORD ordered[],int pair[]);

#endif