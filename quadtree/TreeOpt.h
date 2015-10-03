#ifndef SMART_OPT_H_
#define SMART_OPT_H_

#include <iostream>

#include "Tour.h"
#include "Quadtree.h"
#include "QuadtreeNode.h"

typedef struct SwapCandidate
{
  Segment* segment1;
  Segment* segment2;
  cost_t swap_cost;
} SwapCandidate;

// This class represents the quadtree 2-opt algorithm.
// It is inherently sequential by design, because the swap_candidate is updated 
// with the best swap candidate as the algorithm progresses.
class TreeOpt
{
public:
  TreeOpt(Quadtree* quadtree__, Tour* tour__) : 
    quadtree_(quadtree__), tour_(tour__)
  {
    swap_candidate_.segment1 = nullptr;
    swap_candidate_.segment2 = nullptr;
    swap_candidate_.swap_cost = 0;
  }
  // This finds and stores the best swap in swap_candidate.
  void FindBestSwap();
  SwapCandidate swap_candidate() { return swap_candidate_; }
  void PrintSwapCandidate();
  // Performs a swap based on what is currently in the swap_candidate_.
  void PerformSwap();
private:
  Quadtree* quadtree_;
  Tour* tour_;
  SwapCandidate swap_candidate_;
  void EvaluateNode(QuadtreeNode* node, Segment& segment);
  void EvaluateImmediateSegments(QuadtreeNode* node, Segment& segment);
};


#endif