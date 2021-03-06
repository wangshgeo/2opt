#include "QuadtreeNode.h"

using namespace std;


QuadtreeNode::QuadtreeNode(QuadtreeNode* parent__, int quadrant__,
  pair<morton_key_type, int>* morton_key_pairs, 
  int morton_key_pairs_count, Tour& tour) : 
  parent_(parent__),
  children_count_(0),
  quadrant_(quadrant__),
  diameter_(0),
  total_point_count_(morton_key_pairs_count),
  total_segment_count_(0),
  max_segment_length_(0)
  // highest_order_(0)
{
  DetermineTreeLevel();

  children_[0] = nullptr;
  children_[1] = nullptr;
  children_[2] = nullptr;
  children_[3] = nullptr;

  DetermineChildren(morton_key_pairs, tour);

  DetermineAveragePointLocations(tour);

  ComputeDiameter(morton_key_pairs, tour);

  ComputeMaxSegmentLength();

  // ComputeHighestOrder();
}

void QuadtreeNode::DetermineTreeLevel()
{
  if(parent_ == nullptr)
  {
    tree_level_ = 0;
  }
  else
  {
    tree_level_ = parent_->tree_level() + 1;
  }
  is_leaf_ = total_point_count_ == 1 or tree_level_ >= MAX_LEVEL - 1;
}

struct pair_search_comparator
{
    bool operator()(pair<morton_key_type, int> pair, size_t size) const
    {
      return pair.first < size;
    }
    bool operator()(size_t size, pair<morton_key_type, int> pair) const
    {
      return size < pair.first;
    }
};

// A recursive step.
// To be called by global root.
void QuadtreeNode::DetermineChildren(
  pair<morton_key_type, int>* morton_key_pairs, Tour& tour)
{
  if( is_leaf_ )
  {
    // Add immediate points.
    for(int i = 0; i < total_point_count_; ++i)
    {
      immediate_points_.push_back( morton_key_pairs[i].second );
    }
  }
  else
  {
    // Get the keys corresponding to the start of each sub-quadrant.
    vector<morton_key_type> quadrant_keys = ExtractLeadingQuadrants(
      morton_key_pairs[0].first, tree_level_);
    pair<morton_key_type, int>* end = morton_key_pairs + total_point_count_;
    
    // cout << "quadrants at level " << tree_level_ << ": " << endl;
    // for(int i = 0; i < 4; ++i) 
    // {
    //   cout << bitset<64>(quadrant_keys[i]).to_string().substr(22) << endl;
    // }


    // Determine the starting point of each quadrant.
    pair<morton_key_type, int>* starts[5];
    starts[0] = morton_key_pairs;
    starts[4] = end;
    for(int i = 1; i < 4; ++i)
    {
      starts[i] = std::lower_bound(starts[i-1], end, quadrant_keys[i], 
        pair_search_comparator() );
    }
    
    // Determine size of each quadrant.
    int sizes[4];
    for(int i = 0; i < 4; ++i) sizes[i] = starts[i+1] - starts[i];

    // Call children constructors on non-empty quadrants!
    for(int i = 0; i < 4; ++i)
    {
      if(sizes[i] > 0)
      {
        children_[i] = new QuadtreeNode(this, i, starts[i], sizes[i], tour);
        ++children_count_;
      }
    }
  }
}

// A recursive step.
// Assumes leaves know immediate points.
void QuadtreeNode::DetermineAveragePointLocations(Tour& tour)
{
  average_point_location_[0] = 0;
  average_point_location_[1] = 0;
  
  if(is_leaf_)
  {  
    // Compute point center.
    int immediate_point_count = static_cast<int>(immediate_points_.size());
    for(int i = 0; i < immediate_point_count; ++i)
    {
      average_point_location_[0] += tour.x(immediate_points_[i]);
      average_point_location_[1] += tour.y(immediate_points_[i]);
    }
    average_point_location_[0] /= immediate_point_count;
    average_point_location_[1] /= immediate_point_count;
  }
  else
  {
    for(int i = 0; i < 4; ++i)
    {
      if( children_[i] != nullptr ) 
      {
        children_[i]->DetermineAveragePointLocations(tour);
      }
    }

    for(int i = 0; i < 4; ++i)
    {
      if( children_[i] != nullptr ) 
      {
        average_point_location_[0] += children_[i]->total_point_count_
          * children_[i]->average_point_location_[0];
        average_point_location_[1] += children_[i]->total_point_count_
          * children_[i]->average_point_location_[1];
      }
    }
    average_point_location_[0] /= total_point_count_;
    average_point_location_[1] /= total_point_count_;
  }
}


void QuadtreeNode::Print(int max_level)
{
  if(tree_level_ <= max_level)
  {
    string tabs("");
    for(int i = 0; i < tree_level_; ++i) tabs+="\t";
    cout << endl;
    cout << tabs << "Tree level: " << tree_level_ << endl;
    cout << tabs << "Quadrant: " << quadrant_ << endl;
    cout << tabs << "Total Number of Points: " << total_point_count_ << endl;
    cout << tabs << "Average Point Center: " << average_point_location_[0]
      << ", " << average_point_location_[1] << endl;
    cout << tabs << "Diameter: " << diameter_ << endl;
    cout << tabs << "Total Number of Segments: " << total_segment_count_ << endl;
    cout << tabs << "Immediate Segments: " << immediate_segments_.size() << endl;
    cout << tabs << "Is leaf: " << is_leaf_ << endl;
    cout << endl;
    for(int i = 0; i < 4; ++i)
    {
      if(children_[i] != nullptr) children_[i]->Print(max_level);
    }
  }
}



void QuadtreeNode::ModifyTotalSegmentCount(int amount)
{
  total_segment_count_ += amount;
  if(parent_ != nullptr) parent_->ModifyTotalSegmentCount(amount);
}


void QuadtreeNode::AddImmediateSegment(Segment* segment)
{ 
  immediate_segments_.push_back(segment);
  ModifyTotalSegmentCount(1);
}

// void QuadtreeNode::DeleteImmediateSegment(segment_container::iterator it)
// { 
//   immediate_segments_.erase(it);
//   ModifyTotalSegmentCount(-1);
// }

void QuadtreeNode::UpdateMaxSegmentLength(cost_t old_maximum)
{
  if( max_segment_length_ == old_maximum )
  {
    max_segment_length_ = 0;
    ComputeMaxSegmentLength();
    if(parent_ != nullptr) parent_->UpdateMaxSegmentLength(old_maximum);
  }
}

void QuadtreeNode::DeleteImmediateSegment(Segment* segment)
{ 
  segment_container::iterator it
    = find(immediate_segments_.begin(), immediate_segments_.end(), segment);
  // cout << "Size:" << endl;  
  // cout << immediate_segments_.size() << endl;
  immediate_segments_.erase(it);
  // cout << immediate_segments_.size() << endl;
  ModifyTotalSegmentCount(-1);
  UpdateMaxSegmentLength(segment->length);
}


// We assume average point location has already been computed.
void QuadtreeNode::ComputeDiameter(
  pair<morton_key_type, int>* morton_key_pairs, Tour& tour)
{
  double max_square = 0;
  for(int i = 0; i < total_point_count_; ++i)
  {
    int city_id = morton_key_pairs[i].second;
    double dx = tour.x(city_id) - average_point_location_[0];
    double dy = tour.y(city_id) - average_point_location_[1];
    double square = dx*dx + dy*dy;
    if (square > max_square) max_square = square;
  }
  diameter_ = 2 * sqrt(max_square);
}

void QuadtreeNode::ComputeMaxSegmentLength()
{
  for(segment_container::iterator it = immediate_segments_.begin();
    it != immediate_segments_.end(); ++it)
  {
    if ((*it)->length > max_segment_length_)
    {
      max_segment_length_ = (*it)->length;
    }
  }
  for(int i = 0; i < 4; ++i)
  {
    if(children_[i] != nullptr)
    {
      if(children_[i]->max_segment_length_ > max_segment_length_)
      {
        max_segment_length_ = children_[i]->max_segment_length_;
      }
    }
  }
}

// void QuadtreeNode::ComputeHighestOrder()
// {
//   for(segment_container::iterator it = immediate_segments_.begin();
//     it != immediate_segments_.end(); ++it)
//   {
//     if ((*it)->highest_order > highest_order_)
//     {
//       highest_order_ = (*it)->highest_order_;
//     }
//   }
//   for(int i = 0; i < 4; ++i)
//   {
//     if(children_[i] != nullptr)
//     {
//       if(children_[i]->highest_order_ > highest_order_)
//       {
//         highest_order_ = children_[i]->highest_order_;
//       }
//     }
//   }
// }
