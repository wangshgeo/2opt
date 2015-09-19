#include <chrono>
#include <algorithm>

#include "include/opt2.h"
#include "include/instance.h"
#include "include/solution.h"

//array version
int iterate2Opt_BestImprovement(const Instance& instance,int*tour);
int iterate2Opt_BestImprovement_OpenMP(const Instance& instance,int*tour);
void translateTour(const int initialTour[], int*tour, int nb_cities);

void copyTour(const int source[], int target[],int nc);
coordType computeTourScore(const Instance& instance,int tour[],int nc);
void hillClimb(const Instance& instance,int*tour);
void randomDoubleBridge(int tour[],int nb_cities);
void randomNBridge(int tour[],int nb_cities,int bridges);
void randomSwaps(int tour[],int nb_cities,int swaps);
inline void swap(int*tour,int i,int j);

void ijfromk(int k,int dim,int pair[2])
{//pair[0] is i, pair[1] is j
	//fprintf(stderr,"ijfromk input k: %d\n",k);
	int buff[2];
	buff[0] = (int)(((-1+sqrtf(1+4*2*k)))/2);//floating point calculation!
	buff[1] = k-((buff[0]*(buff[0]+1))>>1);
	//fprintf(stderr,"preliminary pair[] = [%d %d]\n",pair[0],pair[1]);
	//pair[0]=pair[1]-2;
	//pair[1] = k-((pair[0]*(pair[0]+1))>>1);
	pair[0] = buff[1]+1;
	pair[1] = buff[0]+2;
	//fprintf(stderr,"final pair[] = [%d %d]\n",pair[0],pair[1]);
	//pair[0]=dim-pair[0]-1;
	//pair[1]=dim-pair[1]-1-pair[0];
	//pair[0]=dim-3-pair[0];
	//pair[1]=dim-2-pair[0];
	//int tmp = pair[0];pair[0]=pair[1];pair[1]=tmp;
}

Solution solve2Opt(const Instance& instance,long cutoffTime,int randomSeed)
{
	srand(randomSeed);

	int nb_cities = instance.getCityCount();
	//CITY cities[nb_cities];
	//translateTour(initialTour,cities,nb_cities);

	//Random Initial Path
	int tour[nb_cities];
	for(int i=0;i<instance.getCityCount();++i) { tour[i] = i; }

	auto start_time = std::chrono::high_resolution_clock::now();
    cutoffTime*=60*1000;
    
    long duration = 0;
    long bestTime = 0;
    int iterations =0;
    coordType tourScore;
    coordType bestScore = computeTourScore(instance,tour,nb_cities);
	int bestTour[nb_cities];
	while(duration<cutoffTime)
	{

		//Peturbation
		randomRestart(tour,nb_cities);
		//randomDoubleBridge(tour,nb_cities);
		//randomNBridge(tour,nb_cities,16);
		//randomSwaps(tour,nb_cities,5);

		//Find local optimum
		hillClimb(instance,tour);

		++iterations;
		tourScore = computeTourScore(instance,tour,nb_cities);
		//cout << "Local Optimum Reached, Score: " << tourScore << ". ";
		if(tourScore < bestScore)
		{
			copyTour(tour,bestTour,nb_cities);
			bestScore = tourScore;
			auto now = std::chrono::high_resolution_clock::now();    
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
			bestTime = duration;
			cout << "Current Best: " << bestScore << ", achieved in " << bestTime << " ms. ";
			cout << "(Outer Iterations: " << iterations << ") ";
			cout << "\n";
		}
	}

	auto now = std::chrono::high_resolution_clock::now();    
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
	Solution sol = Solution(bestTour,instance,duration,1);
	return sol;
}
void hillClimb(const Instance& instance,int*tour)
{
		int iterations=0;
    	int improvements = 1;
		while(improvements)
		{
			//improvements = iterate2Opt_BestImprovement(instance,tour);
			improvements = iterate2Opt_BestImprovement_OpenMP(instance,tour);
			++iterations;
		}
		//cout << "Inner iterations: " << iterations << ".\n";
}
coordType computeTourScore(const Instance& instance,int tour[],int nc)
{
	coordType score=0;
	for(int i=0;i<nc-1;++i)
	{
			score+=instance.getDistance(tour[i],tour[i+1]);
	}
	score+=instance.getDistance(tour[nc-1],tour[0]);
	return score;
}
void copyTour(const int source[], int target[],int nc)
{
	for(int i=0;i<nc;++i)
	{
		target[i] = source[i];
	}
}
void randomRestart(int tour[],int nb_cities)
{
	random_shuffle(tour,tour+nb_cities);
}
void randomDoubleBridge(int tour[],int nb_cities)
{//double bridge random edge switch
	int c1,c2,c3,c4;
	c1=1+rand()%(nb_cities/4);
	c2=c1+1+rand()%(nb_cities/4);
	c3=c2+1+rand()%(nb_cities/4);
	c4=c3+1+rand()%(nb_cities-c3-3);
	swap(tour,c1,c2);
	swap(tour,c3,c4);
}
void randomNBridge(int tour[],int nb_cities,int bridges)
{//double bridge random edge switch
	int c1,c2;
	int prev=0;
	for(int i=0;i<bridges-1;++i)
	{
		c1 = prev+1+rand()%(nb_cities/bridges/2);
		c2 = c1+1+rand()%(nb_cities/bridges/2);
		swap(tour,c1,c2);
		prev=c2;
	}
	c1=prev+1+rand()%(nb_cities/bridges/2);
	c2=c1+1+rand()%(nb_cities-c1-3);
	swap(tour,c1,c2);
}
void randomSwaps(int tour[],int nb_cities,int swaps)
{//double bridge random edge switch
	for(int i=0;i<swaps;++i)
	{
		int c1 = rand()%(nb_cities-3)+1;
		swap(tour,c1,c1+1+rand()%(nb_cities-3-c1));
	}
}

inline coordType checkSwap(const Instance& instance,const int tour[],int i,int j)
{
	coordType dold=instance.getDistance(tour[i-1],tour[i])+instance.getDistance(tour[j],tour[j+1]);
	coordType dnew=instance.getDistance(tour[i-1],tour[j])+instance.getDistance(tour[i],tour[j+1]);
	return dnew-dold;
}
inline void swap(int*tour,int i,int j)
{
	int length = j-i+1;
	//#pragma omp parallel
	{ 	
		//#pragma omp parallel for
		for(int k=0;k<(length>>1);++k)
		{
			int tmp = tour[j-k];
			tour[j-k]=tour[i+k];
			tour[i+k]=tmp;
		}
	}
	/*
	//An alternate way to do it.
	int buffer[length];
	for(int k=0;k<length;++k)
	{
		buffer[k]=tour[j-k];
	}
	for(int k=0;k<length;++k)
	{
		tour[k] = buffer[k];
	}
	*/
}

int iterate2Opt_BestImprovement(const Instance& instance,int*tour)
{//this version uses only int*tour
	//finds and performs the best swap.
	/**
	Check reversal of cities between i and j, inclusive.
	Thus j-i should be at least 1
	Switching (i,j) is the same as (j,i)
	Check if reversal needs to be made by comparing:
	(i-1 to i) + (j to j+1) (original)
	versus
	(i-1 to j) + (j+1 to i) (swapped)
	**/

	//the main loop; iterate over all possible edge swaps
	int nc = instance.getCityCount();
	int besti=-1;
	int bestj=-1;
	coordType delta,bestDelta=0;
	for(int i=1;i<nc-2;++i)
	{
		for(int j=i+1;j<nc-1;++j)
		{
			delta = checkSwap(instance,tour,i,j);
			if(delta<bestDelta)//delta: the more negative the better
			{
				besti=i;
				bestj=j;
				bestDelta=delta;
			}
		}
	}
	//rearrange the tour to reflect the edge swap
	int improved=0;
	if(bestDelta<0)
	{
		swap(tour,besti,bestj);
		improved=1;
	}
	//return the flag for improvement
	return improved;
}
typedef struct bestDeltaOMP {
	int bestj;
	coordType bestDelta;
} BESTDELTAOMP;
int iterate2Opt_BestImprovement_OpenMP(const Instance& instance,int*tour)
{//this version uses only int*tour, for enabling openmp
	//finds and performs the best swap.
	/**
	Check reversal of cities between i and j, inclusive.
	Thus j-i should be at least 1
	Switching (i,j) is the same as (j,i)
	Check if reversal needs to be made by comparing:
	(i-1 to i) + (j to j+1) (original)
	versus
	(i-1 to j) + (j+1 to i) (swapped)
	**/

	//the main loop; iterate over all possible edge swaps
	int nc = instance.getCityCount();
	BESTDELTAOMP besti[nc];

	for(int i=0;i<nc;++i)
	{
		besti[i].bestDelta=0;
	}
	int totalChecks = ((nc-3)*(nc-2))/2;
	#pragma omp parallel
	{ 	
		#pragma omp for
		for(int k=0;k<totalChecks;++k)
		{
			int i = ((int)((-1+sqrt(1+4*2*k))))>>1;
			int j = k-((i*(i+1))>>1);
			i=nc-3-i;
			j=nc-2-j;
			//cout << " (" << i << ", " << j << ") " << "\n";
			//cin.ignore();
			coordType delta;
			delta = checkSwap(instance,tour,i,j);
			if(delta<besti[i].bestDelta)//delta: the more negative the better
			{
				besti[i].bestj=j;
				besti[i].bestDelta=delta;
			}
		}
	}
	
/*
	//#pragma omp parallel
	{ 	
		//Non-serialized indices.
		#pragma omp parallel for
		for(int i=1;i<nc-2;++i)
		{
			coordType delta;
			besti[i].bestj=0;
			besti[i].bestDelta=0;
			for(int j=i+1;j<nc-1;++j)
			{
				delta = checkSwap(instance,tour,i,j);
				if(delta<besti[i].bestDelta)//delta: the more negative the better
				{
					besti[i].bestj=j;
					besti[i].bestDelta=delta;
				}
				//cout << " (" << i << ", " << j << ") ";
			}
			//cout << "\n";
			//cin.ignore();
		//cout << omp_get_num_threads() << "\n";
		}
	}
*/

	//get the global best
	coordType bestDelta=0;
	int bi=-1;
	int bj=-1;
	for(int i=0;i<nc;++i)
	{
		if(besti[i].bestDelta<bestDelta)
		{
			bestDelta=besti[i].bestDelta;
			bi=i;
			bj=besti[i].bestj;
		}
	}
	//rearrange the tour to reflect the edge swap
	int improved=0;
	if(bestDelta<0)
	{
		swap(tour,bi,bj);
		improved=1;
	}
	//return the flag for improvement
	return improved;
}
BESTIMPROVEMENT iterate2Opt_cudaCompare(const Instance& instance,const int tour[])
{//this version simply returns minimum difference and the index at which it occurs.
	//for comparison purposes with gpu.
	//finds and performs the best swap.
	/**
	Check reversal of cities between i and j, inclusive.
	Thus j-i should be at least 1
	Switching (i,j) is the same as (j,i)
	Check if reversal needs to be made by comparing:
	(i-1 to i) + (j to j+1) (original)
	versus
	(i-1 to j) + (j+1 to i) (swapped)
	**/

	//the main loop; iterate over all possible edge swaps
	int nc = instance.getCityCount();
	int totalChecks = ((nc-3)*(nc-2))/2;
	coordType bestDifference=0;
	int minK=-1;
	int besti=-1,bestj=-1;
	{
		for(int i=1;i<nc-2;++i)
		{
			for(int j=i+1;j<nc-1;++j)
			{
				//for(int k=0;k<totalChecks;++k)
				//{
					//int i = ((int)((-1+sqrt(1+4*2*k))))>>1;
					//int j = k-((i*(i+1))>>1);
					//i=nc-3-i;
					//j=nc-2-j;
				coordType delta = checkSwap(instance,tour,i,j);
				if(delta<bestDifference)//delta: the more negative the better
				{
					bestDifference = delta;
					//minK = k;
					int ki = i-1;
					int kj = j-2;
					minK = ki+((kj*(kj+1))>>1);
					besti = i;
					bestj = j;
				}
			}
		}
	}
	cerr << "best i,j: " << besti << "," << bestj << "\n"; 
	BESTIMPROVEMENT ret = {minK,bestDifference};
	return ret;
}

void translateTour(const int initialTour[], int*tour, int nb_cities)
{//Translate from list<int> to int[]
	for (int i=0;i<nb_cities;++i) {
    	tour[i] = initialTour[i];
	}
	//for (int i=0;i<nb_cities;++i) {	cout << tour[i] << ", ";	} cout << "\n";
}
