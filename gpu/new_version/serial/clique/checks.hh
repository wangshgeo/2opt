#pragma once

#include "types.hh"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

void check_valid_tour(const dtype* x_original, const dtype* y_original, const int nc,
	const dtype* x, const dtype* y);

int compute_tour_length(const dtype*x, const dtype*y, const int nc);