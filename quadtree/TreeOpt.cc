#include "TreeOpt.h"

void TreeOpt::EvaluateImmediateSegments(QuadtreeNode* node, Segment& segment)
{
  if(node->immediate_segment_count() > 0)
  {
    segment_container* immediate_segments = node->immediate_segments();
    for(segment_container::iterator it = immediate_segments->begin(); 
      it < immediate_segments->end(); ++it)
    {
      if(not tour_->adjacent_segments(segment, *(*it)))
      {
        cost_t swap_cost = tour_->SwapCost(segment, *(*it));
        if(swap_cost < swap_candidate_.swap_cost)
        {
          swap_candidate_.segment1 = &segment;
          swap_candidate_.segment2 = *it;
          swap_candidate_.swap_cost = swap_cost;
        }
      }
    }
  }
}

// Recursive
// We assume swap_candidate is already filled with meaningful values.
void TreeOpt::EvaluateNode(QuadtreeNode* node, Segment& segment)
{
  // If node has any segments to compare to.
  if(node->total_segment_count() > 0)
  {
    // Use MAC criterion.
    double max_old_cost = segment.length + node->diameter();
    double dx = abs(segment.center_x - node->average_point_location(0));
    double dy = abs(segment.center_y - node->average_point_location(1));
    if (dx < max_old_cost and dy < max_old_cost)
    {
      double center_distance = sqrt(dx*dx + dy*dy);
      if(center_distance < max_old_cost)
      {
        // Now we must compute and check the possible swaps in this node!
        
        // Check through immediate segments.
        EvaluateImmediateSegments(node, segment);

        // Check through other nodes.
        for(int i = 0; i < 4; ++i)
        {
          if(node->children(i) != nullptr)
          {
            EvaluateNode(node->children(i), segment);
          }
        }
      }
    }
  }
}

void TreeOpt::FindBestSwap()
{
  for(int i = 0; i < tour_->cities(); ++i)
  {
    EvaluateNode(quadtree_->root(),*(tour_->segment(i)));
  }
}