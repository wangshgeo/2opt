#ifndef QUADTREE_SERIAL_H_
#define QUADTREE_SERIAL_H_

// C library headers
#include <cstdio>
#include <cstdlib>
#include <cmath>

// C++ library headers
#include <algorithm>

// Project-specific headers
#include "types.h"
#include "Node.h"
#include "morton_serial.h"

Node* construct_quadtree_serial(Node* parent, int level_index, 
	const std::pair<mtype,int>* morton_keys, const int size, 
	const int current_level, const dtype*x, const dtype*y);

void destroy_quadtree_serial(Node* current_node);

void 
best_improvement_quadtree( int* i_best, int* j_best, cost_t *cost_best,
	Node** best_node, int* best_segment_index,
	const cost_t* segment_lengths, 
	const dtype* center_x, const dtype* center_y,
	const dtype *x, const dtype *y,	const int n, 
	Node* tree,	int*map );


#endif // QUADTREE_SERIAL_H_