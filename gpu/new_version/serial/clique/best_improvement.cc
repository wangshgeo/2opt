#include "best_improvement.hh"

cost_t 
segment_length(const int k, const dtype* x, const dtype* y, const int n)
{
	int next = (k == n-1) ? 0 : k+1;
	dtype dx = x[next] - x[k];
	dtype dy = y[next] - y[k];
	return ( (cost_t) sqrt(dx*dx + dy*dy));
}

cost_t 
total_new_length(const int i, const int j,
	const dtype* x, const dtype* y, const int n)
{
	int inext = (i == n-1) ? 0 : i+1;
	int jnext = (j == n-1) ? 0 : j+1;
	dtype dx0 = x[j] - x[i];
	dtype dy0 = y[j] - y[i];
	dtype dx1 = x[jnext] - x[inext];
	dtype dy1 = y[jnext] - y[inext];
	return  ( (cost_t) sqrt(dx0*dx0 + dy0*dy0)) + ( (cost_t) sqrt(dx1*dx1 + dy1*dy1));
}

void 
best_improvement ( int* i_best, int* j_best, cost_t *cost_best,
	const dtype *x, const dtype *y, const int n )
{
	*cost_best = 0;
	*i_best = -1;
	*j_best = -1;
	for(int i=0;i<n-2;++i)
	{
		for(int j=i+2;j<n;++j)
		{
			cost_t cost_current = total_new_length(i,j,x,y,n)
				- (cost_t) segment_length(i,x,y,n) - (cost_t) segment_length(j,x,y,n); 
			if(cost_current < *cost_best)
			{
				*cost_best = cost_current;
				*i_best = i;
				*j_best = j;
			}
		}
	}
}