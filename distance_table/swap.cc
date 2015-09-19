#include "swap.hh"



void 
swap(const int i, const int j, 
	dtype* x, dtype* y)
{//WARNING! i and j refer to the FIRST point in the segments to be switched.
	flip<dtype>(x,i,j);
	flip<dtype>(y,i,j);
}