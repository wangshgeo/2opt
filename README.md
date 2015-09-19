This repository contains C++ implementations of the 2-Opt heuristic for the 
Euclidean traveling salesman problem (TSP).

# Introduction

If we think of a TSP tour as a sequence of segments that connect cities, a 
2-Opt iteration swaps 2 segments such that the resulting tour is still valid 
(a single closed loop) and the cost of the tour is lower than before the swap.

Successive 2-Opt iterations will eventually result in an locally optimal tour, 
meaning that there will remain no more 2-Opt swaps to perform. Restarting the 
problem from a different initial tour can result in different locally optimum 
solutions.
 
2-Opt is a popular heuristic because it is simple and effective.

# Variants

This repository includes several implementation variants:

1. naive: recompute all distances every iteration.
2. distance_table: store all possible distances in a table, and use this when 
  performing the 2-Opt checks. On my machine on usa13509.tsp, this runs about 
  4x faster than the naive version.
3. gpu: a CUDA GPU version.
4. quadtree: an experimental algorithm utilizing a quadtree in a similar way to 
  the Barnes-Hut algorithm for n-body problems, but tailored to segments.


# Running the Programs

Compilation can be done by simply running 'make' in the main source directory 
for each version. The executable then takes in the filename of the tsp problem 
in the TSPLIB format. See: 
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/