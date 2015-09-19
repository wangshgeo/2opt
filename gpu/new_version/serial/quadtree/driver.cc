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
#include "checks.h"
#include "input.h"
#include "best_improvement.h"
#include "swap.h"
#include "morton_serial.h"
#include "Node.h"
#include "quadtree_serial.h"
#include "segment_serial.h"

#define ITERATIONS 1 // Maximum number of 2-opt iterations.

int main(int argc, char ** argv)
{
  
  int random_seed = 0;
  srand(random_seed);

  int n = 10000;
  dtype *x, *y;

  std::cout << "\nInput checks:\n";
  if(argc >= 2)
  {
    n = getCityCount(argv[1]);
    fprintf(stdout, "\tCity count: %d\n", n);
    x = new dtype[n];
    y = new dtype[n];
    fill_coordinates_2D(argv[1], x, y, n);
    shuffle_cities(x,y,n);
  }
  else
  {
    fprintf(stdout, "\tCity count: %d\n", n);
    x = new dtype[n];
    y = new dtype[n];
    int max = 100000;
    for(int i=0;i<n;++i)
    {
      x[i] = ( (dtype) ( rand() % max ) );
      y[i] = ( (dtype) ( rand() % max ) );
    }
  }
  std::cout << std::endl;

  // The map allows us to modify the tour, without modifying the descriptive
  // data structures, such as x, y, point_morton_pairs.
  // map[i] is a key that corresponds to the ith segment in the tour.
  // The key is used on the descriptive data structures.
  // We can simply flip the map, and thereby flip all other data structures.
  int* map = new int[n];
  for(int i=0;i<n;++i) map[i] = i;

  //Generate the quadtree.
  dtype xmin,xmax,ymin,ymax;
  reduce(x,n,&xmin,&xmax);
  reduce(y,n,&ymin,&ymax);
  morton_key_type* point_morton_pairs = new morton_key_type[n];
  make_morton_keys_serial(point_morton_pairs, x, y, n, xmin, xmax, ymin, ymax);
  Node* tree = construct_quadtree_serial(
    NULL, // We are not inputing a head, since we are retrieving the head.
    -1, // The head is not a child, so its child index is -1.
    point_morton_pairs,
    n, 
    0, // Current level of root is 0.
    x, 
    y
  );

  // Allocate and initialize data structures.
  cost_t* segments = new cost_t[n];
  compute_segment_lengths(x, y, n, segments);
  dtype* segment_center_x = new dtype[n];
  dtype* segment_center_y = new dtype[n];
  compute_segment_centers(segment_center_x,segment_center_y,x,y,n);
  mtype* point_morton_keys = new mtype[n];
  ordered_point_morton_keys(point_morton_pairs, point_morton_keys, n);

  // Populate the quadtree with segments.
  insert_segments(tree, point_morton_keys, n);
  
  // Distance table for reference implementation.
  cost_t* dtable = new cost_t[n*n];
  fill_distance_table(dtable,x,y,n);

  // Tree checks.
  // print_quadtree(tree, "");
  // write_tour(x,y,n);

  long double original_best_improvement_time = 0;
  long double quadtree_best_improvement_time = 0;
  long double flip_time = 0;
  Timer timer;

  fprintf(stdout,"Running best-improvement search...\n");
  for(int i =0; i<ITERATIONS;++i)
  {
    std::cout << "\tIteration " << i+1 << " / " << ITERATIONS << std::endl;

    int i_best_original=0,j_best_original=0;
    int i_best_quadtree=0,j_best_quadtree=0;
    cost_t cost_original;
    cost_t cost_quadtree;
    Node* best_node;
    int best_segment_index;//index in best_node segments (container).

    timer.start();
    best_improvement(&i_best_original,&j_best_original,&cost_original,x,y,n,dtable,map);
    fprintf(stdout, "\tOriginal best: %d %d %f\n", i_best_original, j_best_original, (dtype) cost_original);
    original_best_improvement_time += timer.stop();

    timer.start();
    best_improvement_quadtree( &i_best_quadtree, &j_best_quadtree, &cost_quadtree,
      &best_node, &best_segment_index,
      segments, segment_center_x, segment_center_y,
      x, y, n, tree, map );
    fprintf(stdout, "\tQuadtree best: %d %d %f\n", i_best_quadtree, j_best_quadtree, (dtype) cost_quadtree);
    quadtree_best_improvement_time += timer.stop();

    bool same_indices = i_best_original == i_best_quadtree and j_best_original == j_best_quadtree;
    bool same_cost = cost_original == cost_quadtree;
    if( not same_indices or not same_cost )
    {
      std::cout << "\tError! The best improvements do not match!\n";
      break;
    }

    timer.start();
    swap(i_best_quadtree, j_best_quadtree, x, y);
    flip<int>(map,i_best_quadtree,j_best_quadtree);
    flip_time += timer.stop();

    fprintf(stdout, "\t\tSwitching %d and %d\n", i_best_quadtree, j_best_quadtree);
    fprintf(stdout, "\t\tIteration %d improvement: %ld\n", i, cost_quadtree);
  }
  std::cout << std::endl;

  std::cout << "Timings:\n";
  std::cout << "\tOriginal time: " << original_best_improvement_time << " s\n";
  std::cout << "\tQuadtree time: " << quadtree_best_improvement_time << " s\n";
  std::cout << "\tFlip time: " << flip_time << " s\n";
  std::cout << std::endl;

  std::cout << "Speedups:\n";
  std::cout << "\tQuadtree over Original Speedup: " << 
    original_best_improvement_time / quadtree_best_improvement_time << std::endl;
  std::cout << std::endl;

  delete[] x;
  delete[] y;
  delete[] point_morton_pairs;
  delete[] segments;
  delete[] segment_center_x;
  delete[] segment_center_y;
  delete[] map;
  delete[] point_morton_keys;
  destroy_quadtree_serial(tree);

  std::cout << "Program finished successfully. Exiting.\n\n";

  return EXIT_SUCCESS;
}