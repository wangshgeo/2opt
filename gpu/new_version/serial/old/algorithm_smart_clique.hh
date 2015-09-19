#pragma once
#include "types.hh"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "solution_advance.hh"

int segment_swap_cost(const dtype*x, const dtype*y, const int nc, 
	const int i, const int j);

void initialize_best(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int *best_keys);

void best_improvement_smart_clique(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int *best_keys, int* best_i,int*best_j);

void update_best_smart_clique(const dtype*x, const dtype*y, const int nc, 
	int* best_costs, int*best_keys,
	const int last_i,const int last_j);

void verify_best_improvement(const dtype*x,const dtype*y,const int nc);
void verify_k_best(const dtype*x,const dtype*y,const int nc, int k);