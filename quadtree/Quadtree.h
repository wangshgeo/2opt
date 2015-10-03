#ifndef QUADTREE_H_
#define QUADTREE_H_

#include <vector>
#include <algorithm>
#include <utility>
#include <bitset>

#include "Tour.h"
#include "MortonKey.h"
#include "QuadtreeNode.h"
#include "Segment.h"

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
  QuadtreeNode* root() { return root_; }
  void Print() { root_->Print(); }
  void InsertSegment(Segment* segment);
private:
  morton_key_type* point_morton_keys_; // point morton keys accessible by point 
    // identifier.
  QuadtreeNode* root_;
  double minimum(double* x, int length);
  double maximum(double* x, int length);
  
  // void InsertTourSegments(Tour& tour);
  vector<int> MergePointMortonKeys(morton_key_type key1, morton_key_type key2);
  void InsertTourSegments(Tour& tour);
  void MakeMortonTour(
  vector< pair<morton_key_type, int> >& morton_key_pairs, Tour& tour);
};




#endif