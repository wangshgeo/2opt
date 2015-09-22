// Author: Robert Lee
// Created: Fall 2015
// This is an implementation of 2D Euclidean Traveling Salesman Problem (TSP)
// 2-Opt using quadtrees on the path segments. This code is designed to achieve
// less than quadratic work complexity for non-approximated 2-Opt. The expected
// work complexity is linearithmic. 

// C library headers
#include <stdio.h>
#include <stdlib.h>

// C++ library headers
#include <algorithm>

// Project-specific headers
#include "Timer.h"
#include "input.h"
#include "best_improvement.h"
#include "swap.h"
// #include "morton_serial.h"
// #include "quadtree_serial.h"
// #include "segment_serial.h"
#include "Instance.h"
#include "Tour.h"
#include "QuadtreeNode.h"

using namespace std;

#define ITERATIONS 1 // Maximum number of 2-opt iterations.

int main(int argc, char ** argv)
{
  if(argc < 2)
  {
    cout << "\nPlease enter the input file name!\n\n";
    return EXIT_SUCCESS;
  }

  string file_name(argv[1]);
  cout << file_name << "\n";

  Instance instance(file_name);

  Tour tour(instance);

  // long double quadtree_best_improvement_time = 0;
  // Timer timer;

  cout << "Running iterations: " << endl;
  for(int i =0; i<ITERATIONS;++i)
  {
    cout << "\tIteration " << i+1 << " / " << ITERATIONS << endl;
  }
  cout << endl;

  cout << "Program finished successfully. Exiting.\n\n";

  return EXIT_SUCCESS;
}