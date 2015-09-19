#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

typedef float dtype;

typedef struct coord
{
	dtype x,y;
} COORD;


#define nc 14

void swapTour(int tour[],int pair[])
{//pair[0] = i, pair[1] = j
	int length = pair[1]-pair[0]+1;
	int swapi = pair[0];
	int swapj = pair[1];
	{ 	
		for(int k=0;k<(length>>1);++k)
		{
			int tmp = tour[swapj-k];
			tour[swapj-k]=tour[swapi+k];
			tour[swapi+k]=tmp;
		}
	}
}
void copyCoords(COORD source[],COORD target[])
{
	for(int i=0;i<nc;++i)
	{
		target[i] = source[i];
	}
}
void swapCoords(COORD tour[],int pair[])
{//pair[0] = i, pair[1] = j
	int length = pair[1]-pair[0]+1;
	int swapi = pair[0];
	int swapj = pair[1];
	{ 	
		for(int k=0;k<(length>>1);++k)
		{
			COORD tmp = tour[swapj-k];
			tour[swapj-k]=tour[swapi+k];
			tour[swapi+k]=tmp;
		}
	}
}
void printTour(int tour[])
{
	for(int i=0;i<nc;++i)
	{
		fprintf(stderr,"%d\t",tour[i]);
	}
	fprintf(stderr,"\n");
}
void initTour(int tour[])
{
	for(int i=0;i<nc;++i)
	{
		tour[i] = i;
	}
}
void randomizeDistances(COORD *ordered)
{
	for(int i=0;i<nc;++i)
	{
		ordered[i].x = (rand())%1000;
		ordered[i].y = (rand())%1000;
	}
}
dtype euc2d_distance(COORD i,COORD j)
{
	dtype dx = i.x-j.x;
	dtype dy = i.y-j.y;
	return round(sqrt(dx*dx+dy*dy));
}
dtype computeTourCost(COORD* ordered)
{
	dtype sum = 0;
	for(int i=0;i<nc-1;++i)
	{
		sum+=euc2d_distance(ordered[i],ordered[i+1]);
	}
	sum+=euc2d_distance(ordered[0],ordered[nc-1]);
	return sum;
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
void printDistances(COORD*ordered)
{
	for(int i=0;i<nc;++i)
	{
		fprintf(stderr,"%d\t",(int)euc2d_distance(ordered[i],ordered[(i+1)%nc]));
	}
	fprintf(stderr,"\n");
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
	//fprintf(stderr,"counter: %d\n",counter);
	//fprintf(stderr,"first cost: %f\n",euc2d_distance(tour[start],tour[start+1]));
	//fprintf(stderr,"last cost: %f\n",euc2d_distance(tour[end-1],tour[end]));
	return sum;
}
int main()
{
	int pair[2] = {2,8};

	int tour[nc];
	int newtour[nc];

	initTour(tour);
	initTour(newtour);
	printTour(tour);
	swapTour(newtour,pair);
	printTour(newtour);

	COORD ordered[nc];
	COORD newordered[nc];

	randomizeDistances(ordered);
	copyCoords(ordered,newordered);
	swapCoords(newordered,pair);

	fprintf(stderr,"Original tour cost: %f\n",computeTourCost(ordered));
	fprintf(stderr,"Swap cost: %f\n",pairSwapCost(ordered,pair));
	fprintf(stderr,"New tour cost: %f\n",computeTourCost(newordered));

	printDistances(ordered);
	printDistances(newordered);

	fprintf(stderr,"Original section cost: %f\n",
		sectionCostFromCoords(ordered,pair));
	fprintf(stderr,"New section cost: %f\n",
		sectionCostFromCoords(newordered,pair));


	return 0;
}