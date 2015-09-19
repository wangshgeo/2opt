#pragma once

// C library headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// C++ library headers
#include <string>

// Project-specific headers
#include "types.h"
#include "Node.h"

void check_valid_tour(const dtype* x_original, const dtype* y_original, const int nc,
	const dtype* x, const dtype* y);

int compute_tour_length(const dtype*x, const dtype*y, const int nc);


void print_quadtree(Node* current_node,std::string tabs);


void write_tour( const dtype* x, const dtype* y, const int n );