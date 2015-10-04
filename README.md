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
  Update: Success! This implementation is an order of magnitude faster than 
  the distance_table implementation (the algorithmic state of the art), and 
  even on the CPU is competitive with Martin Burtscher's GPU implementation!  

Quadtree Performance:  
1. usa13509_morton.tsp: ~25x faster than distance_table on my machine.  
2. d18512_morton.tsp: ~108x faster than distance_table on my machine. 

So we see that the quadtree implementation has truly sub-quadratic complexity.

Quadtree Improvements:  
1. Tree updates of filter quantities.

# Running the Programs

Compilation can be done by simply running 'make' in the main source directory 
for each version. The executable then takes in the filename of the tsp problem 
in the TSPLIB format. See: 
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/