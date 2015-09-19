#include "solution_advance.hh"


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

void apply_coordinate_change(dtype*x, dtype*y, const int i, const int j)
{//WARNING! i and j refer to the FIRST point in the segments to be switched.
	flip<dtype>(x,i,j);
	flip<dtype>(y,i,j);
}
