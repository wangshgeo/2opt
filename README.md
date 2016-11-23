This repository contains C++ implementations of the 2-Opt heuristic for the symmetric traveling salesman problem (TSP).

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

1. oldstyle/naive: recompute all distances every iteration.
2. oldstyle/distance_table: store all possible distances in a table, and use this when
  performing the 2-Opt checks. On my machine on usa13509.tsp, this runs about
  4x faster than the naive version.
3. gpu: a CUDA GPU version.
4. quadtree: an experimental algorithm utilizing a quadtree in a similar way to
  the Barnes-Hut algorithm for n-body problems, but tailored to segments.
  Update: Success! This implementation is an order of magnitude faster than
  the distance_table implementation (the algorithmic state of the art) on
  usa13509.tsp, and even on my desktop CPU is competitive with Martin
  Burtscher's GPU implementation peak!

Quadtree performance on my machine:

1. usa13509_morton.tsp: ~25x (parallel) faster than distance_table (serial).
2. d18512_morton.tsp: ~108x (parallel) faster than distance_table (serial).
3. pla33810_morton.tsp: ~544x (parallel) faster than naive (serial). Naive was
  used because distance_table became unresponsive due to memory limits.
4. pla85900_morton.tsp: ~1448x (parallel) faster than naive (serial). Naive was
  used because distance_table became unresponsive due to memory limits.
5. To my surprise, I was able to run the 'world' TSP tour
  (http://www.math.uwaterloo.ca/tsp/world/index.html) at a mere 2 seconds per
  best-improvement step on my desktop machine. This is astronomically faster
  than the standard best-improvement algorithm (it would take too long for me
  to care to test). It runs 15x faster on my desktop CPU than Martin
  Burtscher's GPU Kepler K40 implementation at its peak (which requires many
  simultaneous independent instances).

The quadtree implementation's sub-quadratic complexity is clearly evident.

# Potential Improvements

1. More filter quantities (to reduce tree traversal).
2. More tree updates of filter quantities.
3. k-opt

# Running the Programs

Compilation can be done by simply running 'make' in the main source directory
for each version. The executable then takes in the filename of the tsp problem
in the TSPLIB format. See:
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
