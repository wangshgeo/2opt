#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>

#include "types.hh"

#include "checks.hh"

int 
swapCost(const dtype *x, const dtype *y, const int i, const int j);

void 
best_improvement ( const dtype *x, const dtype *y, const int nc,
	int *best_k, int *best_delta);


void ijfromk(const int k, int *i, int *j);

void check_best_improvement(const dtype *x, const dtype *y, const int nc);
