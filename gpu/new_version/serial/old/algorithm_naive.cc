#include "types.hh"
#include "algorithm_naive.hh"
#include <cmath>

int euc2d_distance(dtype ix, dtype iy, dtype jx, dtype jy)
{
	dtype dx = ix-jx;
	dtype dy = iy-jy;
	return round(sqrt(dx*dx+dy*dy));
}

void ijfromk(const int k, int *i, int *j)
{
	int buff[2];
	buff[0] = (int)(((-1+sqrtf(1+4*2*k)))/2);
	buff[1] = k-((buff[0]*(buff[0]+1))>>1);
	*i = buff[1]+1;
	*j = buff[0]+2;
}
void check_best_improvement(const dtype *x, const dtype *y, const int nc)
{
    int best_k;
    int best_delta;
	best_improvement(x,y,nc,&best_k,&best_delta);

	// fprintf (stdout, "Sequential best k, best delta: %d, %f\n",
		// best_k, best_delta);
	
    // int i,j;
    // ijfromk(best_k,&i,&j);
	// int best_delta_check = swapCost(x,y,i,j);
 //    fprintf(stdout,"(k): %d\n",best_k);
 //    fprintf(stdout,"(i,j): %d,%d\n",i,j);
	// fprintf(stdout,"Swap cost improvement: %d\n",best_delta_check);
}

int 
swapCost(const dtype *x, const dtype *y, const int i, const int j)
{
	dtype ciprevx = x[i-1];
	dtype ciprevy = y[i-1];
	dtype cix = x[i];
	dtype ciy = y[i];
	dtype cjx = x[j];
	dtype cjy = y[j];
	dtype cjnextx = x[j+1];
	dtype cjnexty = y[j+1];
	int now = euc2d_distance(ciprevx,ciprevy,cix,ciy)+euc2d_distance(cjx,cjy,cjnextx,cjnexty);
	int then = euc2d_distance(ciprevx,ciprevy,cjx,cjy)+euc2d_distance(cix,ciy,cjnextx,cjnexty);
	return round(then-now);
}

void 
best_improvement ( const dtype *x, const dtype *y, const int nc,
	int *best_k, int *best_delta)
{
	/**
	Check reversal of cities between i and j, inclusive.
	Thus j-i should be at least 1
	Switching (i,j) is the same as (j,i)
	Check if reversal needs to be made by comparing:
	(i-1 to i) + (j to j+1) (original)
	versus
	(i-1 to j) + (j+1 to i) (swapped)
	**/
	// int totalChecks = ((nc-3)*(nc-2))/2;
	int besti=-1,bestj=-1;
	*best_delta = 0;
	*best_k = -1;
	{
		for(int i=1;i<nc-2;++i)
		{
			for(int j=i+1;j<nc-1;++j)
			{
				int delta = swapCost(x,y,i,j);
				if(delta<*best_delta)
				{
					*best_delta = delta;
					besti = i;
					bestj = j;
				}
			}
		}
	}
	int ki = besti-1;
	int kj = bestj-2;
	*best_k = ki+((kj*(kj+1))>>1);
}