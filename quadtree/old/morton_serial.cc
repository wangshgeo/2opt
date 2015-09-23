#include "morton_serial.h"

btype reinterpret_float_as_int(dtype f)
{
	btype* new_int = reinterpret_cast<btype*>(&f);
	return *new_int;
}

mtype interleave_ints(btype x,btype y)
{
	mtype result = 0;
	// std::cout << sizeof(btype) << std::endl;
	// std::bitset<sizeof(btype)*8> xb(x);
	// std::bitset<sizeof(btype)*8> yb(y);
	// std::cout << "Interleaving " << xb << ", " << yb  << " = ";
	for(int ii = sizeof(btype)*8-1; ii >= 0; --ii)
	{
		result |= (x >> ii) & 1;
		result <<= 1;
		result |= (y >> ii) & 1;
		if(ii != 0)
		{
 			result <<= 1;
		}
	}
	// std::bitset<sizeof(mtype)*8> mb(result);
	// std::cout << mb << std::endl;
	return result;
}

mtype get_morton_key(dtype x, dtype y, dtype max_range, dtype x0, dtype y0)
{
	// btype x_b = reinterpret_float_as_int(x);
	// btype y_b = reinterpret_float_as_int(y);
	btype x_b = (btype) ( ( (1 << MAX_LEVEL) * (x-x0)) / max_range );
	// std::cout << x*(1 << MAX_LEVEL)/domain_size << " vs. " << x_b << std::endl;
	btype y_b = (btype) ( ( (1 << MAX_LEVEL) * (y-y0)) / max_range );
	mtype morton_key = interleave_ints(x_b,y_b);
	return morton_key;
}

void make_morton_keys_serial(morton_key_type* morton_keys,
	const dtype*x,const dtype*y,const int n,
	dtype xmin,dtype xmax,dtype ymin,dtype ymax)
{
	dtype dx = xmax - xmin;
	dtype dy = ymax - ymin;
	dtype max_range = (dx < dy) ? dy : dx;
	
	dtype mff = MORTON_FUDGE_FACTOR*max_range;
	xmin -= mff;
	ymin -= mff;
	max_range += 2 * mff;

	//Embarrassingly parallelizable!
	for(int i=0;i<n;++i)
	{
		morton_keys[i].first = get_morton_key(x[i],y[i],max_range,xmin,ymin);
		morton_keys[i].second = i;
	}
	//Parallelizable!
	std::sort(morton_keys,morton_keys+n);
}


void reduce(const dtype* x, const int n,
	dtype* min, dtype* max)
{
	*min = 0;
	*max = 0;
	for(int i=0;i<n;++i)
	{
		if(x[i] < *min)	*min = x[i];
		if(x[i] > *max) *max = x[i]; 
	}
}

