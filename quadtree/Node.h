#pragma once

// C library headers
#include <cstdlib>
#include <cstdio>
#include <cstddef>

// C++ library headers
#include <vector>

// Project-specific headers
#include "types.h"

typedef std::vector<int> leaf_container;//can contain segment or point ids.

class Node
{
private:
	Node* parent;
	//Child-dependent variables.
	Node* child[4];//pointers to child nodes (null pointers if no child node). Indices (in binary) correspond to Morton ordering.
	int p;//points under this node.
	dtype diameter;//max point deviation distance from center of mass.
	dtype center_of_mass[2];//center of mass of all points under this node.
	leaf_container points;//only leaf nodes shall have points in this container.
	//Child-independent variables.
 	int level;//tree level (root = 0).
	int level_index;//The Morton order of this node relative to siblings (0-3). For the root, this is -1.
	//Properties that change.
	int s;//segments under this node (and only this node).
	leaf_container segments;//If this has elements, this node is a leaf node. They are in Morton order, unless the tree has been merged before.
	int total_s;//total segments under this node and all child nodes.
public:
	//Constructor
	Node(Node* parent_, int level_, int level_index_) : 
		parent(parent_), p(0), 
		diameter(0), level(level_), level_index(level_index_), s(0), total_s(0)
	{
		child[0] = NULL;
		child[1] = NULL;
		child[2] = NULL;
		child[3] = NULL;
		center_of_mass[0] = 0;
		center_of_mass[1] = 0;
	}
	//Setters
	void setChild(Node* child_, int quadrant) { child[quadrant] = child_; }
	void setCenterOfMass(dtype x, dtype y) { center_of_mass[0] = x;center_of_mass[1] = y; }
	void setP(int p_) { p=p_; }
	void setDiameter(dtype radius_) { diameter = 2*radius_; }
	//Getters
	int getLevel() { return level; }
	int getLevelIndex() { return level_index; }
	Node* getChild(int quadrant) { return child[quadrant]; }
	leaf_container* getPoints() { return &points; }
	leaf_container* getSegments() { return &segments; }
	dtype* getCenterOfMass() { return center_of_mass; }
	int getP() { return p; }
	int getS() { return s; }
	int getTotalS() { return total_s; }
	Node* getParent() { return parent; }
	dtype getDiameter() { return diameter; }
	//
	Node* get_extreme_child(int direction);
	Node* remove_empty_nodes(int caller_id);
	bool compute_center_of_mass(const dtype*x,const dtype*y);
	void recompute_center_of_mass(const dtype*x, const dtype*y);
	void remove_extreme_point(int direction);
	void addSegment(int segment) { segments.push_back(segment);++s; }
	void addPoint(int point) { points.push_back(point); }
	void increment_total_s() { total_s = total_s + 1; }
	void deleteSegment(int segment);
};


void remove_extreme_point_and_readjust(Node* root, const int direction, 
	const dtype*x, const dtype*y);
