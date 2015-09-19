#pragma once

// C library headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// C++ library headers
#include <iostream>
#include <string>
#include <sstream>

// Project-specific headers
#include "types.h"
#include "checks.h"


void 
best_improvement ( int* i_best, int* j_best, cost_t *cost_best,
	const dtype *x, const dtype *y, const int n, cost_t* dtable, int*map );


void fill_distance_table(cost_t* dtable, const dtype* x, const dtype* y, const int n);