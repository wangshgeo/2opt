#include "algorithm_smart_clique.hh"

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

int segment_swap_cost(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j)
{
	int diff = abs(i - j);
	if((diff<=1) or (diff == nc-1))
	{
		// fprintf(stdout, "Error! i and j are one apart!, %d, %d\n",i,j);
		return 0;
	}
	// int nexti = ();
	int current = segment_length(x,y,nc,i) + segment_length(x,y,nc,j);
	dtype dxi = x[j] - x[i];
	dtype dyi = y[j] - y[i];
	int nexti = (i<nc-1)?i+1:0;
	int nextj = (j<nc-1)?j+1:0;
	dtype dxj = x[nextj] - x[nexti];
	dtype dyj = y[nextj] - y[nexti];
	int s1 = dxi*dxi + dyi*dyi;
	int s2 = dxj*dxj + dyj*dyj;
	int d1 = sqrt(s1);
	if(d1 > current) return 0;
	int d2 = sqrt(s2);
	if(d2 > current) return 0;
	// if (s1 <= s2) return 0;
	int crossed = d1 + d2;
	return crossed - current;
}

void compute_k_best(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int* best_keys, const int k)
{//find best pairing for segment k.
	best_costs[k] = 0;
	best_keys[k] = -1;
	for(int i=0;i<k-1;++i)
	{
		int check = segment_swap_cost(x,y,nc,i,k);
		if(check < best_costs[k])
		{
			best_costs[k] = check;
			best_keys[k] = i;
		}
	}
}

void initialize_best(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int *best_keys)
{
	for(int k=0;k<nc;++k)
	{
		compute_k_best(x,y,nc,best_costs,best_keys,k);
	}
}

void best_improvement_smart_clique(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int *best_keys, int* best_i,int*best_j)
{
	int best_best_cost = 0;
	for(int k=0;k<nc;++k)
	{
		if(best_costs[k] < best_best_cost)
		{
			// fprintf(stdout, "New best found at %d,%d, with check value of %d.\n",
			// i,best_i_key[i],best_i_cost[i]);
			// std::cin.ignore();
			best_best_cost = best_costs[k];
			if(k < best_keys[k])
			{
				*best_i = k;
				*best_j = best_keys[k];
			}
			else
			{
				*best_j = k;
				*best_i = best_keys[k];
			}
			// fprintf("%d,%d,%d\n",best_best_i_cost,best_best_i_j,best_best_i_i);
			// std::cin.ignore()
		}
	}
	// fprintf(stdout,"(i,j): %d,%d\n",*best_i,*best_j);
	// fprintf(stdout,"Swap cost improvement: %d\n",best_cost);
	// fprintf(stdout,"%d, %d, %d\n",*best_i,*best_j,best_best_cost);
	if(best_best_cost >= 0)
	{
		fprintf(stdout,"No more good moves found! Exiting.\n");
		exit(0);
	}
}

template <class T>
void flip(T* a, const int i,const int j)
{
	int range = (j-i) >> 1;
	// fprintf(stdout, "%d\n",range);
	for(int k=0;k<range;++k)
	{
		int first = i + k + 1;
		int second = j - k;
		// fprintf(stdout, "swp %d\t%d\n",first,second);
		T temp = a[first];
		a[first] = a[second];
		a[second] = temp;
	}
}
void flip_lists_smart_clique(int*best_costs, int*best_keys, const int nc,
	const int i, const int j)
{
	flip<int>(best_costs,i,j);
	flip<int>(best_keys,i,j);
	// for(int k=0;k<nc;++k)
	// {//update key values to reflect flips.
	// 	bool above_i = best_keys[k] > i;
	// 	bool below_j = best_keys[k] < j;
	// 	bool in_flip_range = (above_i and below_j);
	// 	if(in_flip_range)
	// 	{
	// 		best_keys[k] = flip_key(best_keys[k],i,j);
	// 	}
	// }
}

inline bool in_range(int k, int i, int j)
{
	if(abs(i-j) <= 1) fprintf(stdout,"Hey you weren't supposed to call this!\n");
	return k >= i and k <=j; 
}

void compute_k_best_range(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int* best_keys, const int k, const int swap_i, const int swap_j)
{//find best pairing for segment k.
	bool key_in_range = in_range(best_keys[k],swap_i,swap_j);
	bool k_in_range = in_range(k,swap_i,swap_j);
	bool ji = swap_j < swap_i;
	if(key_in_range or k_in_range or ji) fprintf(stdout,"Hey! You weren't supposed to call this!\n");
	int top = (k < swap_j)? k-2:swap_j;
	for(int i=swap_i;i<=top;++i)
	{
		int diff = abs(swap_j-swap_i);
		if(diff > 1 and diff < nc-1)
		{
			int check = segment_swap_cost(x,y,nc,i,k);
			if(check < best_costs[k])
			{
				best_costs[k] = check;
				best_keys[k] = i;
			}
		}
	}
}
void update_best_smart_clique(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int*best_keys,
	const int last_i,const int last_j)
{
	// apply_coordinate_change(x,y,last_i,last_j);
	flip_lists_smart_clique(best_costs,best_keys,nc,last_i,last_j);
	if(last_i >= last_j)
	{
		fprintf(stdout, "Error! last_i >= last_j! %d, %d\n",last_i,last_j);
		exit(0);
	}
	for(int k=0;k<nc;++k)
	{//update every segment.
		bool key_in_range = in_range(best_keys[k],last_i,last_j);
		bool k_in_range = in_range(k,last_i,last_j);
		if(key_in_range or k_in_range)
		{//recompute bests for this segment. Recomputation could be avoided by using buffers.
			// fprintf(stdout,"recomputing bests for k = %d!\n",k);
			compute_k_best(x, y, nc, best_costs, best_keys, k);
			// fprintf(stdout, "new recomputed best: %d, %d, %d\n",best_costs[k],best_keys[k],k);
		}
		else
		{//otherwise, account for new reversed segments!
			compute_k_best_range(x, y, nc, best_costs, best_keys, k, last_i, last_j);
		}
	}
}

int segment_swap_cost_naive(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j)
{
	int diff = abs(i - j);
	if((diff<=1) or (diff == nc-1))
	{
		// fprintf(stdout, "Error! i and j are one apart!, %d, %d\n",i,j);
		return 0;
	}
	int currenti = segment_length(x,y,nc,i);
	int currentj = segment_length(x,y,nc,j);
	return segment_crossed_lengths(x,y,nc,i,j) - currenti - currentj;
}
void verify_best_improvement(const dtype*x,const dtype*y,const int nc)
{
	int best_cost = 0;
	int best_i = 0;
	int best_j = 0;
	for(int i=0;i<nc;++i)
	{
		for(int j=0;j<nc;++j)
		{
			int diff = abs(j-i);
			if (diff > 1 and diff < nc-1)
			{
				int check =	segment_swap_cost_naive(x,y,nc,i,j);
				if(check < best_cost)
				{
					best_i = i;
					best_j = j;
					best_cost = check;
				}
			}
		}
	}
	fprintf(stdout, "Verified best improvement: %d, %d, %d\n",best_i,best_j,best_cost);
}
void verify_k_best(const dtype*x,const dtype*y,const int nc, int k)
{
	int best_cost = 0;
	int best_i = 0;
	int best_j = 0;
	for(int i=0;i<nc;++i)
	{
		int diff = abs(k-i);
		if (diff > 1 and diff < nc-1)
		{
			int check =	segment_swap_cost_naive(x,y,nc,i,k);
			if(check < best_cost)
			{
				best_i = i;
				best_j = k;
				best_cost = check;
			}
		}
	}
	fprintf(stdout, "Verified best improvement for %d: %d, %d, %d\n",k, best_i,best_j,best_cost);
}
