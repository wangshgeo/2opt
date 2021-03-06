#ifndef QUADTREE_NODE_H_
#define QUADTREE_NODE_H_

#include <vector>
#include <utility>
#include <algorithm>
#include <bitset>

#include "Tour.h"
#include "MortonKey.h"
#include "Segment.h"

typedef std::vector<int> id_container;// can contain segment or point 
	// identifiers.
typedef std::vector<Segment*> segment_container;


// This is designed to be called recursively.
class QuadtreeNode
{
public:
	QuadtreeNode(QuadtreeNode* parent__, int quadrant__,
		std::pair<morton_key_type, int>* morton_key_pairs, 
		int morton_key_pairs_count, Tour& tour);
	int tree_level() { return tree_level_; }
	~QuadtreeNode()
	{
		for(int i = 0; i < 4; ++i)
		{
			if (children_[i] != nullptr) delete children_[i];
		}
	}
	QuadtreeNode* children(int index) { return children_[index]; }
	int total_segment_count() { return total_segment_count_; }
	segment_container* immediate_segments() { return &immediate_segments_; }
	int immediate_segment_count() { return immediate_segments_.size(); }
	double diameter() { return diameter_; }
	double average_point_location(int index)
	{
		return average_point_location_[index];
	}
	void AddImmediateSegment(Segment* segment);
	void Print(int max_level = MAX_LEVEL);
	void DeleteImmediateSegment(Segment* segment);
	int quadrant() { return quadrant_; }
	cost_t max_segment_length() { return max_segment_length_; }
	void set_max_segment_length(cost_t new_length) 
		{ max_segment_length_ = new_length; }
private:
	// Tree location information.
	QuadtreeNode* parent_;
	int children_count_; // number of children currently held in children_.
	QuadtreeNode* children_[4];//pointers to child nodes (null pointers if no 
		// child node). Index corresponds to Morton order quadrant.
 	int tree_level_; // tree level (root = 0).
	int quadrant_;// The Morton order of this node relative to siblings (0-3). 
		//For the root, this is -1.
	bool is_leaf_;

	// Point information.
	double average_point_location_[2]; // average location of all points under 
		// this node and its children.
	double diameter_; // diameter of the circle, with center at the average point 
		// location, that encompasses all points under this node.
	id_container immediate_points_; // points under this node only (not 
		// children) (only leaf nodes shall have points in this container).
	int total_point_count_; // points under this node.

	// Segment information.
	segment_container immediate_segments_; // segments under this node only (not 
		// children).
	int total_segment_count_; // total segments under this node and all child 
		// nodes.
	cost_t max_segment_length_; // The maximum segment length of all segments 
		// under this node (both immediate and child segments).
	// int highest_order_; // The maximum order index of all segments under 
		// this node. Used to filter searches. 

	void DetermineTreeLevel();
	void DetermineChildren(pair<morton_key_type, int>* morton_key_pairs, 
		Tour& tour);
	void DetermineAveragePointLocations(Tour& tour);
	void ModifyTotalSegmentCount(int amount);
	void DeleteImmediateSegment(segment_container::iterator it);
	void ComputeDiameter(pair<morton_key_type, int>* morton_key_pairs, 
		Tour& tour);
	void ComputeMaxSegmentLength();
	void UpdateMaxSegmentLength(cost_t old_maximum);
};

#endif