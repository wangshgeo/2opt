#ifndef MORTON_SERIAL_H_
#define MORTON_SERIAL_H_

// C++ library headers
#include <iostream>
#include <bitset>
#include <algorithm>
#include <utility>

// Project-specific headers
#include "types.h"

typedef std::pair<mtype,int> morton_key_type;

mtype interleave_ints(btype x,btype y);

void make_morton_keys_serial(morton_key_type* morton_keys,
	const dtype*x,const dtype*y,const int n,
	dtype xmin,dtype xmax,dtype ymin,dtype ymax);

void reduce(const dtype* x, const int n,
	dtype* min, dtype* max);

#endif // MORTON_SERIAL_H_