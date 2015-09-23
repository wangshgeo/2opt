#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <bitset>

#include "Tour.h"
#include "MortonKey.h"
#include "QuadtreeNode.h"

// The purpose of this class is to handle the generation and sorting of Morton 
// keys, and quadtree construction.
class Quadtree
{
public:
  Quadtree(Tour& tour);
  ~Quadtree()
  {
    delete[] point_morton_keys_;
    delete root_;
  }
  void Print() { root_->Print(); }
private:
  morton_key_type* point_morton_keys_; // point morton keys accessible by point 
    // identifier.
  QuadtreeNode* root_;
  double minimum(double* x, int length);
  double maximum(double* x, int length);
  
};




#endif