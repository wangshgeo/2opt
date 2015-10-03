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
	const dtype *x, const dtype *y, const int n, cost_t* dtable, int*map )
{
	// cost_t best_new_cost;
	// cost_t best_old_cost;
	*cost_best = 0;
	*i_best = -1;
	*j_best = -1;
	for(int i=0;i<n-2;++i)
	{
		for(int j=i+2;j<n;++j)
		{
			int ii = map[i];
			int jj = map[j];
			int nextii = map[i+1];
			int nextjj = ( j == n-1 ) ? map[0] : map[j+1];
			cost_t new_cost = dtable[ii*n + jj] + dtable[nextii*n + nextjj];
			cost_t old_cost = dtable[ii*n + nextii] + dtable[jj*n + nextjj];
			cost_t cost_current = new_cost - old_cost;
			// cost_t std_new = total_new_length(i,j,x,y,n);
			// cost_t std_old = segment_length(i,x,y,n) + segment_length(j,x,y,n);
			// if(new_cost != std_new)
			// {
			// 	fprintf(stdout, "new_cost does not match! %ld vs %ld\n", new_cost, std_new);
			// 	fprintf(stdout, "i = %d segment compare: %ld vs %ld\n", i, dtable[ii*n + ii+1], segment_length(i,x,y,n));
			// 	fprintf(stdout, "j = %d segment compare: %ld vs %ld\n", j, dtable[jj*n + nextjj], segment_length(j,x,y,n));
			// 	std::cin.ignore();
			// }
			// if(old_cost != std_old)
			// {
			// 	fprintf(stdout, "old_cost does not match! %ld vs %ld\n", old_cost, std_old);
			// 	std::cin.ignore();
			// }
			
			// if(i == 3499 and j == 4271)
			// {
			// 	fprintf(stdout, "\n");
			// 	fprintf(stdout, "\t%d and %d\n", i, j);
			// 	fprintf(stdout, "\t%d and %d\n", ii, jj);
			// 	fprintf(stdout, "\t\tnew_cost: %ld\n", new_cost);
			// 	fprintf(stdout, "\t\told_cost: %ld\n", old_cost);
			// 	fprintf(stdout, "\t\t\told segments: %ld, %ld\n", 
			// 		dtable[ii*n + nextii], dtable[jj*n + nextjj]);
			// 	fprintf(stdout, "\t\tcost: %ld\n", cost_current);
			// }
			if(cost_current < *cost_best)
			{
				*cost_best = cost_current;
				*i_best = i;
				*j_best = j;
				// best_new_cost = new_cost;
				// best_old_cost = old_cost;
			}
		}
	}

	// fprintf(stdout, "\t%d and %d\n", *i_best, *j_best);
	// fprintf(stdout, "\t\tnew_cost: %ld\n", best_new_cost);
	// fprintf(stdout, "\t\told_cost: %ld\n", best_old_cost);
	// fprintf(stdout, "\t\tcost: %ld\n", *cost_best);
}



cost_t point_distance(const int n1, const int n2, const dtype*x, const dtype*y)
{
	dtype dx = x[n1]-x[n2];
	dtype dy = y[n1]-y[n2];
	return ( (cost_t) sqrt(dx*dx+dy*dy) );
}

void fill_distance_table(cost_t* dtable, const dtype* x, const dtype* y, const int n)
{
	for(int i=0;i<n;++i)
	{
		for(int j=0;j<n;++j)
		{
			dtable[i*n + j] = point_distance(i,j,x,y); 
		}
	}
}