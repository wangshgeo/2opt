#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "Instance.hh"
#include "timer.hh"
#include "checks.hh"

#include "solution_advance.hh"

#include "algorithm_naive.hh"
// #include "algorithm_segment.hh"
#include "algorithm_smart_clique.hh"

#define ITERATIONS 10
#define REFRESH 1
#define BUFFERS 1

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
	random_shuffle(h_initialTour,h_initialTour+nc);
	fprintf(stdout,"Done.\n");
	
	fprintf(stdout,"Initializing (Host) Coordinates... ");
	dtype *h_x,*h_y;
	h_x = instance.getX();
	h_y = instance.getY();
	dtype* ordered_x = new dtype[nc];
	dtype* ordered_y = new dtype[nc];
	for(int i=0;i<nc;++i)
	{
		ordered_x[i]=h_x[h_initialTour[i]];
		ordered_y[i]=h_y[h_initialTour[i]];
	}
	for(int i=0;i<nc;++i)
	{
		h_x[i]=ordered_x[i];
		h_y[i]=ordered_y[i];
	}
	delete[] h_initialTour;
	fprintf(stdout,"Done.\n");

	//Segment algorithm allocations
	// int* best_i_cost = new int[nc];
	// int* best_j_cost = new int[nc];
	// int* best_i_key = new int[nc];
	// int* best_j_key = new int[nc];
	int* best_costs = new int[nc*BUFFERS];
	int* best_keys = new int[nc*BUFFERS];
	int* segment_lengths = new int[nc];
	for(int i=0;i<nc-1;++i)
	{
		dtype dx = h_x[i+1]-h_x[i];
		dtype dy = h_y[i+1]-h_y[i];
		segment_lengths[i] = round(sqrt(dx*dx) + sqrt(dy*dy));
	}
	dtype dx = h_x[0]-h_x[nc-1];
	dtype dy = h_y[0]-h_y[nc-1];
	segment_lengths[nc-1] = round(sqrt(dx*dx) + sqrt(dy*dy));
	fprintf(stdout,"Initializing best lists... ");
	// initialize_best(h_x,h_y,nc,
	// 	best_i_cost,best_j_cost,best_i_key,best_j_key);
	initialize_best(h_x,h_y,nc,
		best_costs,best_keys);
	fprintf(stdout,"Done.\n");
	//end
	long double old_time = 0;
	long double new_time = 0;
	long double flip_time = 0;
	long double update_time = 0;

	fprintf(stdout,"Running best-improvement search...\n");
	for(int i =0; i<ITERATIONS;++i)
	{
		if((i % REFRESH) == 0)
		{
			check_valid_tour(ordered_x,ordered_y,nc,h_x,h_y);
			int tour_length = compute_tour_length(h_x,h_y,nc);
			fprintf(stdout, "%d\t%d\n",i,tour_length);
		}

		stopwatch_t old_algorithm = stopwatch_start();
		check_best_improvement(h_x,h_y,nc);
		old_time += stopwatch_stop(old_algorithm);

		int best_i,best_j;

		stopwatch_t new_algorithm = stopwatch_start();
		// best_improvement_segment(h_x,h_y,nc,
		// 	best_i_cost,best_j_cost,best_i_key,best_j_key,
		// 	&best_i,&best_j);
		best_improvement_smart_clique(h_x,h_y,nc,
			best_costs,best_keys, &best_i,&best_j);
		new_time += stopwatch_stop(new_algorithm);

		fprintf(stdout, "calculated best: %d\t%d\t%d\t%d\n",best_i,best_j,best_costs[best_i],best_costs[best_j]);
		verify_best_improvement(h_x,h_y,nc);
		// verify_k_best(h_x,h_y,nc,36);
		// fprintf(stdout, "my test: %d\t%d\t%d\n",best_costs[36],best_keys[36],segment_swap_cost(h_x,h_y,nc,36,80));
		// int k = 36;
		// fprintf(stdout,"%d: %f,%f\n",k,h_x[k],h_y[k]);
		// fprintf(stdout,"%d: %f,%f\n",k,h_x[k+1],h_y[k+1]);
		// int j = 79;
		// fprintf(stdout,"%d: %f,%f\n",j,h_x[j],h_y[j]);
		// fprintf(stdout,"%d: %f,%f\n",j,h_x[j+1],h_y[j+1]);
		// int ii = 71;
		// fprintf(stdout,"%d: %f,%f\n",ii,h_x[ii],h_y[ii]);
		// fprintf(stdout,"%d: %f,%f\n",ii,h_x[ii+1],h_y[ii+1]);

		stopwatch_t flip = stopwatch_start();
		apply_coordinate_change(h_x,h_y,best_i,best_j);
		flip_time += stopwatch_stop(flip);

		stopwatch_t new_algorithm_update = stopwatch_start();
		update_best_smart_clique(h_x,h_y, nc, 
			best_costs, best_keys, best_i,best_j);
		update_time += stopwatch_stop(new_algorithm_update);

		// update_best_i(h_x, h_y, nc, 
		// 	best_i_cost, best_i_key,
		// 	best_i, best_j);
		// update_best_j(h_x, h_y, nc, 
		// 	best_i_cost, best_i_key,
		// 	best_i, best_j);
		// apply_list_change(best_i_cost,best_j_cost,
		// 	best_i_key,best_j_key,
		// 	best_i,best_j);
	}
	fprintf(stdout,"\nDone.\n");

	fprintf(stdout, "Speedup: %Le\n",
		(old_time+flip_time) / (new_time+flip_time+update_time) );

	delete[] ordered_x;
	delete[] ordered_y;
	delete[] best_costs;
	delete[] best_keys;
	delete[] segment_lengths;
	// delete[] best_j_cost;
	// delete[] best_i_key;
	// delete[] best_j_key;

	return EXIT_SUCCESS;
}