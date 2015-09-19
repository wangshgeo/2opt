#pragma once
#include "types.hh"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

inline int segment_length(const dtype*x, const dtype*y, const int nc, 
	const int key);

inline int segment_crossed_lengths(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j);

inline int segment_swap_cost(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j);

void compute_best_i(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_i_key, const int i);

void compute_best_j(const dtype*x, const dtype*y, const int nc, 
	int* best_j_cost, int* best_j_key, const int j);

void initialize_best(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_j_cost, int *best_i_key, int* best_j_key);

void best_improvement_segment(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int* best_j_cost, int *best_i_key, int* best_j_key,
	int* best_i,int*best_j);

void update_best_i(const dtype*x, const dtype*y, const int nc, 
	int* best_i_cost, int *best_i_key,
	const int last_i,const int last_j);

void update_best_j(const dtype*x, const dtype*y, const int nc, 
	int* best_j_cost, int *best_j_key,
	const int last_i,const int last_j);