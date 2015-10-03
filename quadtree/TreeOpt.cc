#include "TreeOpt.h"

using namespace std;

void TreeOpt::EvaluateImmediateSegments(QuadtreeNode* node, Segment& segment, 
  SwapCandidate& candidate)
{
  if(node->immediate_segment_count() > 0)
  {
    segment_container* immediate_segments = node->immediate_segments();
    for(segment_container::iterator it = immediate_segments->begin(); 
      it < immediate_segments->end(); ++it)
    {
      // bool adjacent = tour_->adjacent_segments(segment, *(*it));
      bool right_order = segment.order < ((*it)->order-1);
      if( right_order )
      {
        cost_t old_cost = segment.length + (*it)->length;
        if( -old_cost < candidate.swap_cost )
        {
          // cost_t swap_cost = tour_->SwapCost(segment, *(*it));
          cost_t swap_cost = tour_->SwapCost(segment, *(*it), old_cost);
          if(swap_cost < candidate.swap_cost)
          {
            // if(segment.order == 4959 or segment.order == 5154)
            // {
            //   // if((*it)->order == 4959 or (*it)->order == 5154)
            //   {
            //     cout << "Hey! We found it! " << (*it)->order << endl;
            //   }
            // }
            candidate.segment1 = &segment;
            candidate.segment2 = *it;
            candidate.swap_cost = swap_cost;
          }
        }
      }
    }
  }
}

// Recursive
// We assume swap_candidate is already filled with meaningful values.
void TreeOpt::EvaluateNode(QuadtreeNode* node, Segment& segment, 
  SwapCandidate& candidate)
{
  // If node has any segments to compare to.
  if(node->total_segment_count() > 0)
  {
    // Use MAC criterion.
    double max_old_cost = segment.length + node->max_segment_length();
    double dx = abs(segment.center_x - node->average_point_location(0));
    double dy = abs(segment.center_y - node->average_point_location(1));
    // if(segment.order == 4959)
    // {
    //   cout << "level, quadrant: " << node->tree_level() << ", " 
    //     << node->quadrant() << endl;
    //   cout << "dx, dy, max_old_cost: " << dx << ", " << dy << ", " 
    //     << max_old_cost 
    //     << endl;
    // }
    double subtraction_term = segment.length + node->diameter();
    if (2*dx-subtraction_term < max_old_cost 
      and 2*dy-subtraction_term < max_old_cost)
    {
      double center_distance = sqrt(dx*dx + dy*dy);
      double min_new_cost = 2*center_distance - subtraction_term;
      if(min_new_cost < max_old_cost)
      {
        // Now we must compute and check the possible swaps in this node!
        
        // Check through immediate segments.
        EvaluateImmediateSegments(node, segment, candidate);

        // Check through other nodes.
        for(int i = 0; i < 4; ++i)
        {
          if(node->children(i) != nullptr)
          {
            EvaluateNode(node->children(i), segment, candidate);
          }
        }
      }
    }
  }
}

void TreeOpt::FindBestSwap()
{
  ResetSwapCandidate();
  #ifdef _OPENMP
    // Parallel version
    int CHUNKS = 2*omp_get_num_procs();
    // cout << "omp_get_num_procs: " << omp_get_num_procs() << endl;
    // cout << "omp_get_max_threads: " << omp_get_max_threads() << endl;
    int elements_per_chunk = ( tour_->cities() + CHUNKS - 1 ) / CHUNKS;
    SwapCandidate* candidates = new SwapCandidate[CHUNKS]();
    #pragma omp parallel for
    for(int c = 0; c < CHUNKS; ++c)
    {
      int start = c*elements_per_chunk;
      int end = (c+1)*elements_per_chunk;
      end = (end < tour_->cities()) ? end : tour_->cities();
      for(int i = start; i < end; ++i)
      {
        EvaluateNode(quadtree_->root(),*(tour_->segment(i)), 
          candidates[c]);
      }
    }
    int best_index = 0;
    cost_t best_cost = 0;
    for(int c = 0; c < CHUNKS; ++c)
    {
      if(candidates[c].swap_cost < best_cost)
      {
        best_index = c;
        best_cost = candidates[c].swap_cost;
      }
    }
    swap_candidate_.swap_cost = candidates[best_index].swap_cost;
    swap_candidate_.segment1 = candidates[best_index].segment1;
    swap_candidate_.segment2 = candidates[best_index].segment2;
    delete[] candidates;
  #else
    //Serial version
    for(int i = 0; i < tour_->cities(); ++i)
    {
      EvaluateNode(quadtree_->root(),*(tour_->segment(i)), swap_candidate_);
    }
  #endif
}

void TreeOpt::PrintSwapCandidate()
{
  bool has_candidate = swap_candidate_.swap_cost < 0;
  if( has_candidate )
  {
    if( swap_candidate_.segment1 == nullptr )
    {
      cout << "\tSegment 1 is a null pointer!" << endl;
    }
    if( swap_candidate_.segment2 == nullptr )
    {
      cout << "\tSegment 2 is a null pointer!" << endl;
    }
    cout << "Swap Candidate:" << endl;
    cout << "\tCost: " << swap_candidate_.swap_cost << endl;
    cout << "\tSegments: " << swap_candidate_.segment1->order; 
    cout << ", " << swap_candidate_.segment2->order << endl;
  }
  else
  {
    cout << "No swap candidate." << endl;
  }
}

Segment* TreeOpt::PrintSegment(int order)
{
  for(int i = 0; i < tour_->cities(); ++i)
  {
    if( tour_->segment(i)->order == order )
    {
      cout << "\tSegment " << order << " length: " 
        << tour_->segment(i)->length << endl;
      cout << "\tNodal diameter: " 
        << tour_->segment(i)->node->diameter() << endl;
      cout << "\tTree level: " 
        << tour_->segment(i)->node->tree_level() << endl;
      cout << "\tCity ids: " << tour_->segment(i)->start_city 
        << ", " << tour_->segment(i)->end_city << endl;
      return tour_->segment(i);
    }
  }
  return nullptr;
}

// Returns true if a swap can be performed.
bool TreeOpt::PerformSwap()
{
  if(swap_candidate_.swap_cost < 0)
  {
    Segment* segment1 = swap_candidate_.segment1;
    Segment* segment2 = swap_candidate_.segment2;
    segment1->node->DeleteImmediateSegment( segment1 );
    segment2->node->DeleteImmediateSegment( segment2 );
    tour_->Swap( *segment1, *segment2 );
    quadtree_->InsertSegment( segment1 );
    quadtree_->InsertSegment( segment2 );
    return true;
  }
  else
  {
    cout << "No more swaps to perform!" << endl;
  }
  return false;
}

void TreeOpt::ResetSwapCandidate()
{
  swap_candidate_.segment1 = nullptr;
  swap_candidate_.segment2 = nullptr;
  swap_candidate_.swap_cost = 0;
}