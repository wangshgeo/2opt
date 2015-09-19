#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

#include "types.hh"
#include "checks.hh"


void 
best_improvement ( int* i_best, int* j_best, cost_t *cost_best,
	const dtype *x, const dtype *y, const int n, cost_t* dtable, int*map );


void fill_distance_table(cost_t* dtable, const dtype* x, const dtype* y, const int n);