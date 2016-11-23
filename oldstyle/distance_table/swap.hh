#pragma once

#include "types.hh"



void
swap(const int i, const int j, 
	dtype* x, dtype* y);



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
