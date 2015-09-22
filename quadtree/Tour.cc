#include "Tour.h"

using namespace std;

void Tour::Swap(Segment& first_deleted, Segment& second_deleted)
{
  int higher = first_deleted.order;
  int lower = second_deleted.order;
  if( higher < lower )
  {
    higher = second_deleted.order;
    lower = first_deleted.order;
  }
  // Reverse.
  for(int i = 0; i < cities_; ++i)
  {
    int order = segments_[i].order;
    if( order > lower and order < higher )
    {
      reverse_segment(segments_[i],
        first_deleted, second_deleted);
    }
  }
  // Delete and construct anew.
  if(first_deleted.order < second_deleted.order) 
  {
    int original_first_end_city = first_deleted.end_city;
    renew_segment(first_deleted, 
      first_deleted.start_city, second_deleted.start_city);
    renew_segment(second_deleted, 
      original_first_end_city, second_deleted.end_city);
  }
  else
  {
    int original_second_end_city = second_deleted.end_city;
    renew_segment(second_deleted, 
      second_deleted.start_city, first_deleted.start_city);
    renew_segment(first_deleted, 
      original_second_end_city, first_deleted.end_city);
  }
  
}

// Called by the constructor for the initial tour.
void Tour::InitializeSegments()
{
  for(int i = 0; i < cities_-1; ++i)
  {
    renew_segment(segments_[i], i, i+1);
    segments_[i].order = i;
  }
  renew_segment(segments_[cities_-1], cities_-1, 0);
  segments_[cities_-1].order = cities_-1;
}

// This is to be called on segments between deleted segments in the event of a 
// swap. The order will change, but not the length.
void Tour::reverse_segment(Segment& interior, 
  Segment& first_deleted, Segment& second_deleted)
{
  int tmp = interior.start_city;
  interior.start_city = interior.end_city;
  interior.end_city = tmp;
  interior.order = first_deleted.order
    + (second_deleted.order - interior.order);
}

cost_t Tour::Cost(int city1, int city2)
{
  double dx = x_[city1] - x_[city2];
  double dy = y_[city1] - y_[city2];
  return (cost_t)(sqrt(dx*dx + dy*dy));
}

// If a segment is being renewed, that means a swap is occuring (and 
// initialization).
// In a swap, the order does not change, but the length does.
void Tour::renew_segment(Segment& segment, 
  int new_start_city, int new_end_city)
{
  segment.start_city = new_start_city;
  segment.end_city = new_end_city;
  segment.length = Cost(new_end_city, new_start_city);
}

cost_t Tour::SwapCost(Segment& s1, Segment& s2)
{
  cost_t old_cost = s1.length + s2.length;
  cost_t new_cost = Cost(s1.start_city, s2.start_city)
    + Cost(s1.end_city, s2.end_city);
  return new_cost - old_cost;
}

bool Tour::adjacent_segments(Segment& s1, Segment& s2)
{
  int order_diff = abs(s1.order - s2.order);
  bool adjacent = order_diff < 2 or order_diff == cities_-1;
  return adjacent;
}

void Tour::SerialCheck()
{
  int best_swap[2] = {-1,-1};
  int best_swap_[2] = {-1,-1};
  cost_t best_delta = (cost_t)0;

  for(int i=0;i<cities_;++i)
  {
    for(int j = i+1; j < cities_; ++j)
    {
      if ( not adjacent_segments(segments_[i], segments_[j]) )
      {
        cost_t delta = SwapCost(segments_[i], segments_[j]);
        if(delta < best_delta)
        {
          best_delta = delta;
          best_swap[0] = segments_[i].order;
          best_swap[1] = segments_[j].order;
          best_swap_[0] = i;
          best_swap_[1] = j;
        }
      }
    }
  }

  cout << "Best cost: " << best_delta << endl;
  cout << "Best swap: " << best_swap[0] << ", " << best_swap[1] << endl;

  Swap(segments_[best_swap_[0]], segments_[best_swap_[1]]);
}

void Tour::Check()
{
  bool *visited = new bool[cities_];
  for(int i = 0; i < cities_; ++i) visited[i] = false;
  for(int i = 0; i < cities_; ++i)
  {
    bool order_found = false;
    for(int j = 0; j < cities_; ++j)
    {
      if(segments_[j].order == i)
      {
        if(i == 0) visited[segments_[j].start_city] = true;
        order_found = true;
        // cout << "Found segment " << i << endl;
        if( (visited[segments_[j].end_city] and i < cities_-1) or 
          (not visited[segments_[j].start_city]))
        {
          cout << "Tour check failed! City visited twice or " << 
            "city was not visited!" << endl;
          delete[] visited;
          return;
        }
        visited[segments_[j].end_city] = true;
        break;
      }
    }
    if(not order_found)
    {
      cout << "Tour check failed! Segment not found." << endl;
      break;
    }
  }
  delete[] visited;
}