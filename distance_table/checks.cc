#include "checks.hh"

void check_valid_tour(const dtype* x_original, const dtype* y_original,
	const int nc,	const dtype* x, const dtype* y)
{
	int* checklist = new int[nc];
	for(int i=0;i<nc;++i) checklist[i] = 0;
	for(int i=0;i<nc;++i)
	{
		for(int j=0;j<nc;++j)
		{
			if(x[i] == x_original[j] and y[i] == y_original[j])
			{
				++checklist[j];
				if(checklist[j] > 2)
				{
					fprintf(stdout,"Error! A city (%d) was repeated! Exiting.\n", j);
					delete[] checklist;
					return;
				}
			}
		}
	}
	for(int i =0; i<nc;++i)
	{
		if(checklist[i] != 1)
		{
			fprintf(stdout, "Error! City %d appeared %d time(s).\n",i,checklist[i]);
			delete[] checklist;
			return;
		}
	}
	delete[] checklist;
}

int compute_tour_length(const dtype*x, const dtype*y, const int nc)
{
	int tour_length = 0;
	for(int i=0;i<nc-1;++i)
	{
		dtype dx = x[i+1] - x[i];
		dtype dy = y[i+1] - y[i];
		tour_length += round(sqrt(dx*dx + dy*dy)); 
	}
	dtype dx = x[0] - x[nc-1];
	dtype dy = y[0] - y[nc-1];
	tour_length += round(sqrt(dx*dx + dy*dy)); 
	return tour_length;
}
