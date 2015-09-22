#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <vector>
#include <algorithm>
#include <utility>

#include "Tour.h"
#include "MortonKey.h"

// The purpose of this class is to handle the generation and sorting of Morton 
// keys, and quadtree construction.
class Quadtree
{
public:
  Quadtree(Tour& tour);
  ~Quadtree()
  {
    delete[] point_morton_keys;
  }
private:
  morton_key_type* point_morton_keys; // point morton keys accessible by point 
    // identifier.
  double minimum(double* x, int length);
  double maximum(double* x, int length);
};




#endif