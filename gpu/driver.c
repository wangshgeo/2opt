#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <cstdio>
#include <ctime>

#include "include/instance.h"
#include "driver.h"
#include "include/solution.h" 
#include "opt2gpu.h"
#include "include/opt2.h"

#include "parameters.h"

dtype
reduceCpu (dtype* h_A, unsigned int N)
{
	int i;
	// dtype ans, ans2;
	double ans;

	ans = 0.0;
	for(i = 0; i < N; i++) {
		ans += h_A[i];
	}

	return ans;
}


void
initTour (int* tour, unsigned int N)
{
	//srand48 (time (NULL));
	for(int i = 0;i < N; i++) {
		tour[i] = i;
		//A[i] = (rand () & 0xFF) / ((dtype) RAND_MAX);
	}
}
void
initArray (dtype* A, unsigned int N)
{
	unsigned int i;
	srand48 (time (NULL));
	for(i = 0;i < N; i++) {
		// A[i] = drand48 ();
		A[i] = (rand () & 0xFF) / ((dtype) RAND_MAX);
	}
}
void swapCoords(COORD ordered[],int pair[])
{//pair[0] = i, pair[1] = j
	int length = pair[1]-pair[0]+1;
	int swapi = pair[0];
	int swapj = pair[1];
	{ 	
		for(int k=0;k<(length>>1);++k)
		{
			COORD tmp = ordered[swapj-k];
			ordered[swapj-k]=ordered[swapi+k];
			ordered[swapi+k]=tmp;
		}
	}
}
dtype euc2d_distance(COORD i,COORD j)
{
	dtype dx = i.x-j.x;
	dtype dy = i.y-j.y;
	return round(sqrt(dx*dx+dy*dy));
}
dtype costFromCoords(COORD *tour,int nc)
{
	dtype sum = 0;
	for(int i=0;i<nc-1;++i)
	{
		sum+=euc2d_distance(tour[i],tour[i+1]);
	}
	sum+=euc2d_distance(tour[nc-1],tour[0]);
	return sum;
}
dtype sectionCostFromCoords(COORD *tour,int pair[])//,dtype compare[],int compareflag)
{
	dtype sum = 0;
	int counter = 0;
	for(int i=pair[0];i<pair[1];++i)
	{
		dtype dd = euc2d_distance(tour[i],tour[i+1]);
		sum+=dd;
		++counter;
	}
	fprintf(stderr,"counter: %d\n",counter);
	fprintf(stderr,"first cost: %f\n",euc2d_distance(tour[pair[0]],tour[pair[0]+1]));
	fprintf(stderr,"last cost: %f\n",euc2d_distance(tour[pair[1]-1],tour[pair[1]]));
	return sum;
}
void printCoords(COORD*tour,int nc)
{
	for(int i=0;i<nc;++i)
	{
		fprintf(stderr,"%f %f\n",tour[i].x,tour[i].y);
	}
}


dtype pairSwapCost(COORD*tour,int pair[])
{
	int i = pair[0];
	int j = pair[1];
	COORD ciprev = tour[i-1];
	COORD ci = tour[i];
	COORD cj = tour[j];
	COORD cjnext = tour[j+1];
	dtype now = euc2d_distance(ciprev,ci)+euc2d_distance(cj,cjnext);
	dtype then = euc2d_distance(ciprev,cj)+euc2d_distance(ci,cjnext);
	return round(then-now);
}
void compareSums(dtype c1[],dtype c2[],int n,int reverse)
{
	dtype sum1=0,sum2=0;
	for(int i =0;i<n;++i)
	{
		int i2 = (reverse)?(n-1-i):i;
		if(c1[i]!=c2[i2])
		{
			fprintf(stderr,"Difference detected at: %d of %d\n",i,n);
		}
		sum1+=c1[i];
		sum2+=c2[i];
	}
	fprintf(stderr,"Recomputed sums: %f and %f\n",sum1,sum2);
}
void compareTours(COORD c1[],COORD c2[],int n,int reverse)
{
	for(int i =0;i<n;++i)
	{
		int i2 = (reverse)?(n-1-i):i;
		if(c1[i].x!=c2[i2].x || c1[i].y!=c2[i2].y)
		{
			fprintf(stderr,"Coordinate difference detected at: %d of %d\n",i,n-1);
		}
	}
}
void compareDistances(COORD c1[],COORD c2[],int n,int reverse)
{
	for(int i =0;i<n-1;++i)
	{
		int i2 = (reverse)?(n-1-i):i;
		int dir2 = (reverse)?-1:1;
		dtype d1 = euc2d_distance(c1[i],c1[i+1]);
		dtype d2 = euc2d_distance(c2[i2],c2[i2+dir2]);
		if(d1!=d2)
		{
			fprintf(stderr,"Distance difference detected at: %d of %d\n",i,n-1);
		}
	}
}
void assessSwap(COORD*before,COORD*after,int nc,int pair[])
{
	int length = pair[1]-pair[0]+1;
	fprintf(stderr,"Checking coords...\n");
	fprintf(stderr,"First part...\n");
	compareTours(before,after,pair[0],0);
	fprintf(stderr,"Middle part...\n");
	compareTours(before+pair[0],after+pair[0],length,1);
	fprintf(stderr,"Last part...\n");
	compareTours(before+pair[1]+1,after+pair[1]+1,nc-pair[1]-1,0);
	fprintf(stderr,"\n");
	//swapped edges
	//compareSums(before+pair[0]-1,after+pair[1]-1,1);
	//compareSums(before+pair[1]-1,after+pair[0]-1,0);
	fprintf(stderr,"Checking distances...\n");
	fprintf(stderr,"First part...\n");
	compareDistances(before,after,pair[0],0);
	fprintf(stderr,"Middle part...\n");
	compareDistances(before+pair[0],after+pair[0],length,1);
	fprintf(stderr,"Last part...\n");
	compareDistances(before+pair[1]+1,after+pair[1]+1,nc-pair[1]-1,0);
	fprintf(stderr,"\n");
}
int main (int argc, char** argv)
{
	//Input arguments:
	//.tsp file name to read
	char*tspfile = argv[1];

	fprintf(stderr,"Reading in the TSP file at: %s\n",tspfile);
	Instance instance(tspfile);
	int nc = instance.getCityCount();
	fprintf(stderr,"City Count: %d\n\n",nc);
	fprintf(stderr,"Done reading tsp input file.\n");

	fprintf(stderr,"Initializing Host and GPU Data...\n");
	/* declare and initialize data */
	//Initialize Tour
	int h_initialTour[instance.getCityCount()];
	initTour (h_initialTour, nc);
	randomRestart(h_initialTour,nc);
	//int *d_initialTour;
	//initCudaTour (&d_initialTour, h_initialTour, nc);
	//Initialize Coordinates (ordered in the tour order)
	//d_coords and ordered store the first point again in the last element to capture all edges.
	COORD*h_coords = instance.getCoords();
	COORD ordered[nc];
	for(int i=0;i<nc;++i) ordered[i]=h_coords[h_initialTour[i]];
	COORD*d_coords;
	initCudaCoords(&d_coords,ordered,nc);
	fprintf(stderr,"Done.\n\n");
	
	//2Opt Runs
	//Sequential test
	if(RUNSERIAL>0)
	{
		std::clock_t start;
	    double duration=0;
	    BESTIMPROVEMENT bi;
	    for(int i=0;i<SERIAL_ITER;++i)
	    {
	    	start = std::clock();
			bi = iterate2Opt_cudaCompare(instance,h_initialTour);
	    	duration += ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	    }
		fprintf (stderr, "Sequential run time: %f ms\n",
			duration/((double)SERIAL_ITER)*1000.0);
		fprintf (stderr, "Sequential min diff, k: %f, %d\n\n",
			bi.difference,bi.k);
		//Swap test
		std::clock_t start2;
	    double duration2=0;
	    //Get i,j to swap
	    int pair[2];
	    ijfromk(bi.k,nc,pair);
	    cerr << "ijfromk: " << pair[0] << "," << pair[1] << endl;
		dtype rsc = pairSwapCost(ordered,pair);
		fprintf(stderr,"Recomputed swap cost: %f\n",rsc);
	    //end i,j retrieval
	    dtype original = costFromCoords(ordered,nc);
		fprintf(stderr,"Cost before swap: %f\n",original);
		//printCoords(ordered,nc);
		dtype c1 = sectionCostFromCoords(ordered,pair);

		int length = pair[1]-pair[0];
		dtype mycompare1[length];
		for(int i=0;i<length;++i)
		{
			int in = pair[0]+i;
			mycompare1[i]=euc2d_distance(ordered[in],ordered[in+1]);
		}


		COORD orderedoriginal[nc];
		for(int i=0;i<nc;++i) orderedoriginal[i]=ordered[i];


	    //timing
    	start2 = std::clock();
		fprintf(stderr,"Swapping %d,%d\n",pair[0],pair[1]);
	    swapCoords(ordered,pair);
    	duration2 = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
    	//end timing



		fprintf (stderr, "Full swap run time: %f ms\n",
			duration2*1000.0);
		dtype newcost = costFromCoords(ordered,nc);
		fprintf(stderr,"Cost after swap: %f\n",newcost);

		dtype mycompare2[length];
		for(int i=0;i<length;++i)
		{
			int in = i+pair[0];
			mycompare2[length-1-i] = 
				euc2d_distance(ordered[in],ordered[in+1]);
		}
		compareSums(mycompare1,mycompare2,length,0);

		//printCoords(ordered,nc);
		dtype costdiff = newcost-original;
		fprintf(stderr,"Difference: %f\n",costdiff);
		dtype c2 = sectionCostFromCoords(ordered,pair);
		fprintf(stderr,"Swap section costs: %f, %f (difference: %f)\n",c1,c2,c1-c2);
		fprintf(stderr,"\n");
		//End swap test

		assessSwap(orderedoriginal,ordered,nc,pair);
	}
	//End sequential test

	
	cudaOpt2 (ordered, d_coords, nc);

	//End 2opt Run
	
	return 0;
}
