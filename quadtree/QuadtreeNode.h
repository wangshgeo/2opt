#ifndef QUADTREE_NODE_H_
#define QUADTREE_NODE_H_

#include <vector>
#include <utility>

#include "MortonKey.h"

typedef std::vector<int> id_container;// can contain segment or point 
	// identifiers.

// This is designed to be called recursively.
class QuadtreeNode
{
public:
	QuadtreeNode(QuadtreeNode* parent__, int quadrant__,
		std::pair<morton_key_type, int>* morton_key_pairs, 
		int morton_key_pairs_count);
	int tree_level() { return tree_level_; }

private:
	// Tree location information.
	QuadtreeNode* parent_;
	int children_count_; // number of children currently held in children_.
	QuadtreeNode* children_[4];//pointers to child nodes (null pointers if no 
		// child node). Index corresponds to Morton order quadrant.
 	int tree_level_; // tree level (root = 0).
	int quadrant_;// The Morton order of this node relative to siblings (0-3). 
		//For the root, this is -1.

	// Point information.
	double average_point_location_[2]; // average location of all points under 
		// this node and its children.
	double diameter_; // diameter of the circle, with center at the average point 
		// location, that encompasses all points under this node.
	id_container immediate_points_; // points under this node only (not 
		// children) (only leaf nodes shall have points in this container).
	int total_point_count_; // points under this node.

	// Segment information.
	id_container immediate_segments_; // segments under this node only (not 
		// children).
	int total_segment_count_; // total segments under this node and all child 
		// nodes.
};

#endif