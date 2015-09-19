#pragma once

// C library headers
#include <cmath>

// C++ library headers
#include <bitset>
#include <iostream>

//Porject-specific headers
#include "morton_serial.h"
#include "Node.h"


// Get path segment lengths.
void compute_segment_lengths(const dtype* x, const dtype* y, const int n, 
  cost_t* segments);

// Get the quadrant of the key at a given level.
mtype get_level_msb( const mtype key, const int level);

void ordered_point_morton_keys( const morton_key_type* morton_pairs, 
  mtype* morton_keys, const int n );


void insert_segment( Node* tree, const mtype* point_morton_keys, 
	const int segment_index, const int next_segment_index );
void delete_segment( Node* tree, const mtype* point_morton_keys, 
	const int segment_index, const int next_segment_index );

void insert_segments( Node* tree, const mtype* point_morton_keys, const int n);

void compute_segment_centers(dtype* segment_center_x, dtype* segment_center_y, 
	const dtype* x, const dtype* y, const int n);