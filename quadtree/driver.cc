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
#include "Instance.h"
#include "Tour.h"
#include "Quadtree.h"
#include "QuadtreeNode.h"
#include "TreeOpt.h"

using namespace std;

#define ITERATIONS 20 // Maximum number of 2-opt iterations.

int main(int argc, char ** argv)
{
  if(argc < 2)
  {
    cout << "\nPlease enter the input file name!\n\n";
    return EXIT_SUCCESS;
  }

  string file_name(argv[1]);
  cout << "Reading " << file_name << "\n";

  Instance instance(file_name);

  Tour tour(instance);

  Quadtree quadtree(tour);

  // tour.OutputFile("world_morton.tsp");

  // quadtree.Print(2);

  TreeOpt solver(&quadtree, &tour);

  long double quadtree_best_improvement_time = 0;
  Timer timer;

  cout << "Running iterations: " << endl << endl;
  int iterations = 1;
  for( ; iterations < ITERATIONS; ++iterations)
  {
    cout << "Iteration " << iterations << " / " << ITERATIONS << endl;
    timer.start();
    solver.FindBestSwap();
    quadtree_best_improvement_time += timer.stop();
    solver.PrintSwapCandidate();
    // Segment* seg1 = solver.PrintSegment(4959);
    // Segment* seg2 = solver.PrintSegment(5154);
    // cout << "cost of 4959, 5154 swap: " << tour.SwapCost(*seg1, *seg2) << endl;
    bool swapped = solver.PerformSwap();

    // tour.SerialCheck();

    //tour.Check();

    cout << endl;
    if(not swapped) break;
  }
  cout << endl;

  cout << "Average FindBestSwap time: " 
    << quadtree_best_improvement_time / iterations << endl;

  tour.OutputFile("iterated_tour.tsp");

  cout << "Program finished successfully. Exiting.\n\n";

  return EXIT_SUCCESS;
}
