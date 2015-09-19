#ifndef __OPT2_H__
#define __OPT2_H__

#include "types.h"
#include "solution.h"
#include "instance.h"

//In a 2-opt switch, 
//a valid tour results when from both original segments,
//the two starting points
//are paired and the two ending points are paired.


typedef struct bestimprovement
{
	int k;//serialized index.
	coordType difference;
} BESTIMPROVEMENT;


Solution solve2Opt(const Instance& instance,long cutoffTime,int randomSeed);
void randomRestart(int tour[],int nb_cities);
BESTIMPROVEMENT iterate2Opt_cudaCompare(const Instance& instance,const int tour[]);
void ijfromk(int k,int dim,int pair[]);

#endif