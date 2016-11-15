#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "Instance.hh"
#include "timer.hh"
#include "checks.hh"

#include "best_improvement.hh"
#include "swap.hh"

#define ITERATIONS 5
#define REFRESH 1

int main(int argc, char ** argv)
{
	if(argc < 2)
	{
		fprintf(stderr,"Please enter an input file!\n");
		return EXIT_FAILURE;
	}

	int random_seed = 0;
	srand(random_seed);

	char* city_file_name = argv[1];
	fprintf(stdout,"Reading file at: %s\n",city_file_name);
	Instance instance(city_file_name,false);
	int nc = instance.getCityCount();
	fprintf(stdout,"City count: %d\n\n",nc);
	fprintf(stdout,"Done reading file.\n");

	fprintf(stdout,"Initializing (Host) Initial Tours... ");
	int *h_initialTour;
	h_initialTour = new int[nc];
	for(int i=0;i<nc;++i) h_initialTour[i] = i;
	// random_shuffle(h_initialTour,h_initialTour+nc);
	fprintf(stdout,"Done.\n");
	
	fprintf(stdout,"Initializing (Host) Coordinates... ");
	dtype *x,*y;
	x = instance.getX();
	y = instance.getY();
	dtype* ordered_x = new dtype[nc];
	dtype* ordered_y = new dtype[nc];
	for(int i=0;i<nc;++i)
	{
		ordered_x[i]=x[h_initialTour[i]];
		ordered_y[i]=y[h_initialTour[i]];
	}
	for(int i=0;i<nc;++i)
	{
		x[i]=ordered_x[i];
		y[i]=ordered_y[i];
	}
	cost_t* dtable = new cost_t[nc*nc];
	fill_distance_table(dtable,x,y,nc);
	int* map = new int[nc];
	for(int i=0;i<nc;++i)
	{
		map[i] = i;
	}
	delete[] h_initialTour;
	fprintf(stdout,"Done.\n");

	long double best_improvement_time = 0;
	long double flip_time = 0;

	fprintf(stdout,"Running best-improvement search...\n");
	for(int i =0; i<ITERATIONS;++i)
	{
		// if((i % REFRESH) == 0)
		// {
		//	check_valid_tour(ordered_x,ordered_y,nc,x,y);
		// 	int tour_length = compute_tour_length(x,y,nc);
		// 	fprintf(stdout, "%d\t%d\n",i,tour_length);
		// }

		int i_best,j_best;
		cost_t cost;

		stopwatch_t best_improvement_watch = stopwatch_start();
		best_improvement(&i_best,&j_best,&cost,x,y,nc,dtable,map);
		best_improvement_time += stopwatch_stop(best_improvement_watch);

		stopwatch_t flip_watch = stopwatch_start();
		swap(i_best, j_best, x, y);
		flip<int>(map,i_best,j_best);
		flip_time += stopwatch_stop(flip_watch);

		fprintf(stdout, "Switching %d and %d\n", i_best, j_best);
		fprintf(stdout, "Iteration %d improvement: %ld\n", i, cost);
	}
	fprintf(stdout,"\nDone.\n");

	fprintf(stdout, "Average best-improvement search time: %Le\n", 
		best_improvement_time / ITERATIONS );

	delete[] ordered_x;
	delete[] ordered_y;
	delete[] dtable;
	delete[] map;

	return EXIT_SUCCESS;
}
