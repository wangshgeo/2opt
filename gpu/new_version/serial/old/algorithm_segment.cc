#include "algorithm_segment.hh"


int flip_key(const int key, const int i, const int j)
{
	if(key <= i or key >= j) 
	{
		fprintf(stdout, "Wrong usage of flip_key()!\n");
		exit(0);
	}
	int return_key = key;
	if(key > i and key < j)
	{
		return_key = j - (key - i);
	}
	return return_key;
}


inline int segment_length(const dtype*x, const dtype*y, const int nc, 
	const int key)
{
	int next = (key<nc-1)?key+1:0;
	dtype dx = x[next] - x[key];
	dtype dy = y[next] - y[key];
	return round(sqrt(dx*dx + dy*dy));
}
inline int segment_crossed_lengths(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j)
{
	dtype dxi = x[j] - x[i];
	dtype dyi = y[j] - y[i];
	int nexti = (i<nc-1)?i+1:0;
	int nextj = (j<nc-1)?j+1:0;
	dtype dxj = x[nextj] - x[nexti];
	dtype dyj = y[nextj] - y[nexti];
	return round(sqrt(dxi*dxi + dyi*dyi)) + round(sqrt(dxj*dxj + dyj*dyj));
}
inline int segment_swap_cost(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j)
{
	int currenti = segment_length(x,y,nc,i);
	int currentj = segment_length(x,y,nc,j);
	return segment_crossed_lengths(x,y,nc,i,j) - currenti - currentj;
}
void compute_best_i(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_i_key, const int i)
{
	best_i_cost[i] = 0;
	best_i_key[i] = -1;
	for(int j=i+2;j<nc;++j)
	{
		int check = segment_swap_cost(x,y,nc,i,j);
		if(check < best_i_cost[i])
		{
			best_i_cost[i] = check;
			best_i_key[i] = j;
		}
	}
}
void compute_best_j(const dtype*x, const dtype*y, const int nc, 
	int* best_j_cost, int* best_j_key, const int j)
{
	best_j_cost[j] = 0;
	best_j_key[j] = -1;
	for(int i = 0;i<j-2;++i)
	{
		int check = segment_swap_cost(x,y,nc,i,j);
		if(check < best_j_cost[j])
		{
			best_j_cost[j] = check;
			best_j_key[j] = i;
		}
	}
}

void initialize_best(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_j_cost, int *best_i_key, int* best_j_key)
{
	for(int i = 0;i<nc-2;++i)
	{
		compute_best_i(x,y,nc,best_i_cost,best_i_key,i);
	}
	for(int j=2;j<nc;++j)
	{
		compute_best_j(x,y,nc,best_j_cost,best_j_key,j);
	}
}

void best_improvement_segment(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_j_cost, int *best_i_key, int* best_j_key,
	int* best_i,int*best_j)
{
	int best_best_i_cost = 0;
	int best_best_j_cost = 0;
	int best_best_i_i = 0;
	int best_best_j_i = 0;
	int best_best_i_j = 0;
	int best_best_j_j = 0;
	for(int i = 0;i<nc-2;++i)
	{
		if(best_i_cost[i] < best_best_i_cost)
		{
			// fprintf(stdout, "New best found at %d,%d, with check value of %d.\n",
			// 	i,best_i_key[i],best_i_cost[i]);
			// std::cin.ignore();
			best_best_i_cost = best_i_cost[i];
			best_best_i_j = best_i_key[i];
			best_best_i_i = i;
			// fprintf("%d,%d,%d\n",best_best_i_cost,best_best_i_j,best_best_i_i);
			// std::cin.ignore()
		}
	}
	for(int j=2;j<nc;++j)
	{
		if(best_j_cost[j] < best_best_j_cost)
		{
			// fprintf(stdout, "New best found at %d,%d, with check value of %d.\n",
			// 	j,best_j_key[j],best_j_cost[j]);
			// std::cin.ignore();
			best_best_j_cost = best_j_cost[j];
			best_best_j_i = best_j_key[j];
			best_best_j_j = j;
		}
	}
	int best_cost = best_best_i_cost;
	*best_i = best_best_i_i;
	*best_j = best_best_i_j;
	if(best_best_i_cost > best_best_j_cost)
	{
		best_cost = best_best_j_cost;
		*best_i = best_best_j_i;
		*best_j = best_best_j_j;
	}
 //    fprintf(stdout,"(i,j): %d,%d\n",*best_i,*best_j);
	// fprintf(stdout,"Swap cost improvement: %d\n",best_cost);
	fprintf(stdout,"%d, %d, %d\n",*best_i,*best_j,best_cost);
}

void update_best(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_j_cost, 
	int *best_i_key, int* best_j_key,
	const int last_i,const int last_j)
{
	for(int i = 0;i<last_j-2;++i)
	{
		if(best_i_key[i] == last_j or i == last_i)
		{
			compute_best_i(x, y, nc, best_i_cost, best_i_key, i);
		}
		else
		{
			int check = segment_swap_cost(x,y,nc,i,last_j);
			if(check < best_i_cost[i])
			{
				best_i_cost[i] = check;
				best_i_key[i] = last_j;
			}
		}
	}
	for(int i = last_j+2;i<nc-2;++i)
	{
		if(best_i_key[i] == last_j or i == last_i)
		{
			compute_best_i(x, y, nc, best_i_cost, best_i_key, i);
		}
		else
		{
			int check = segment_swap_cost(x,y,nc,i,last_j);
			if(check < best_i_cost[i])
			{
				best_i_cost[i] = check;
				best_i_key[i] = last_j;
			}
		}
	}
	for(int j = 2;j<last_i-2;++j)
	{
		if(best_j_key[j] == last_i or j == last_j)
		{
			compute_best_j(x, y, nc, best_j_cost, best_j_key, j);
		}
		else
		{
			int check = segment_swap_cost(x,y,nc,last_i,j);
			if(check < best_j_cost[j])
			{
				best_j_cost[j] = check;
				best_j_key[j] = last_i;
			}
		}
	}
	for(int j = last_i+2;j<nc;++j)
	{
		if(best_j_key[j] == last_i or j == last_j)
		{
			compute_best_j(x, y, nc, best_j_cost, best_j_key, j);
		}
		else
		{
			int check = segment_swap_cost(x,y,nc,last_i,j);
			if(check < best_j_cost[j])
			{
				best_j_cost[j] = check;
				best_j_key[j] = last_i;
			}
		}
	}
}

void update_best_i(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int *best_i_key,
	const int last_i,const int last_j)
{//assumes x and y have been swapped at last_i and last_j, but best costs and keys are not.
	for(int i=0;i<nc-2;++i)
	{
		bool last_is_best = ((last_i == best_i_key[i]) or (last_j == best_i_key[i]));
		if(last_is_best)
		{//Redo best check.
			fprintf(stdout, "Redoing best check for i = %d!\n",i);
			compute_best_i(x, y, nc, best_i_cost, best_i_key, i);
			fprintf(stdout, "%d, %d, %d\n",i,);
		}
		else
		{//otherwise fill in holes.
			bool best_in_flip_range = ((last_i < best_i_key[i]) or (last_j > best_i_key[i]));
			if(best_in_flip_range)
			bool last_i_in_row = last_i >= (i+2);
			if(last_i_in_row)
			{//last_i hole.
				int check = segment_swap_cost(x,y,nc,i,last_i);
				if(check < best_i_cost[i])
				{
					best_i_cost[i] = check;
					best_i_key[i] = last_i;
				}
			}
			//last_j hole.
			int check = segment_swap_cost(x,y,nc,i,last_j);
			if(check < best_i_cost[i])
			{
				best_i_cost[i] = check;
				best_i_key[i] = last_j;
			}
		}
	}
}
void update_best_j(const dtype*x, const dtype*y, const int nc, 
	int* best_j_cost, int *best_j_key,
	const int last_i,const int last_j)
{//assumes x and y have been swapped at last_i and last_j, but best costs and keys are not.
	compute_best_j(x, y, nc, best_j_cost, best_j_key, last_i);
	compute_best_j(x, y, nc, best_j_cost, best_j_key, last_j);
}