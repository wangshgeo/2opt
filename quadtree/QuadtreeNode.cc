#include "QuadtreeNode.h"

QuadtreeNode::QuadtreeNode(QuadtreeNode* parent__, int quadrant__,
	std::pair<morton_key_type, int>* morton_key_pairs, 
	int morton_key_pairs_count) : 
	parent_(parent__),
	children_count_(0),
	quadrant_(quadrant__),
	diameter_(0),
	total_point_count_(0),
	total_segment_count_(0)
{
	if(parent_ == nullptr)
	{
		tree_level_ = 0;
	}
	else
	{
		tree_level_ = parent_->tree_level() + 1;
	}
	children_[0] = nullptr;
	children_[1] = nullptr;
	children_[2] = nullptr;
	children_[3] = nullptr;
	average_point_location_[0] = 0;
	average_point_location_[1] = 0;
}











// void remove_extreme_point_and_readjust(Node* root, const int direction, 
// 	const dtype*x, const dtype*y);
// 	//Setters
// 	void setChild(Node* child_, int quadrant) { child[quadrant] = child_; }
// 	void setCenterOfMass(dtype x, dtype y) { center_of_mass[0] = x;center_of_mass[1] = y; }
// 	void setP(int p_) { p=p_; }
// 	void setDiameter(dtype radius_) { diameter = 2*radius_; }
// 	//Getters
// 	int getLevel() { return level; }
// 	int getLevelIndex() { return level_index; }
// 	Node* getChild(int quadrant) { return child[quadrant]; }
// 	leaf_container* getPoints() { return &points; }
// 	leaf_container* getSegments() { return &segments; }
// 	dtype* getCenterOfMass() { return center_of_mass; }
// 	int getP() { return p; }
// 	int getS() { return s; }
// 	int getTotalS() { return total_s; }
// 	Node* getParent() { return parent; }
// 	dtype getDiameter() { return diameter; }
// 	//
// 	Node* get_extreme_child(int direction);
// 	Node* remove_empty_nodes(int caller_id);
// 	bool compute_center_of_mass(const dtype*x,const dtype*y);
// 	void recompute_center_of_mass(const dtype*x, const dtype*y);
// 	void remove_extreme_point(int direction);
// 	void addSegment(int segment) { segments.push_back(segment);++s; }
// 	void addPoint(int point) { points.push_back(point); }
// 	void increment_total_s() { total_s = total_s + 1; }
// 	void deleteSegment(int segment);






// void Node::deleteSegment(int segment)
// { 
// 	for(int i=0;i<s;++i)
// 	{
// 		if(segments[i] == segment) segments.erase( segments.begin() + i );
// 	}
// }

// void Node::remove_extreme_point(int direction)
// {//direction: 0 for left, >0 for right.
// 	if(direction == 0)
// 	{
// 		points.erase(points.begin());
// 	}
// 	else
// 	{
// 		points.pop_back();
// 	}
// }

// bool Node::compute_center_of_mass(const dtype*x,const dtype*y)
// {//Returns true if center of mass changed, false if not.
// 	bool changed = false;
// 	dtype xm_sum = 0;
// 	dtype ym_sum = 0;
// 	int p_sum = 0;
// 	for(int i=0;i<4;++i)
// 	{
// 		Node* c = child[i];
// 		if(c != NULL)
// 		{
// 			int cp = c->getP();
// 			p_sum += cp;
// 			dtype* com = c->getCenterOfMass();
// 			xm_sum += cp * com[0];
// 			ym_sum += cp * com[1];
// 		}
// 	}
// 	for(unsigned int i=0;i<points.size();++i)
// 	{
// 		++p_sum;
// 		xm_sum += x[points[i]];
// 		ym_sum += y[points[i]];
// 	}
// 	dtype new_xm = (p_sum > 0) ? (xm_sum / p_sum) : 0;
// 	dtype new_ym = (p_sum > 0) ? (ym_sum / p_sum) : 0;
// 	if ( (p != p_sum) or (new_xm != center_of_mass[0]) or (new_ym != center_of_mass[1]) )
// 	{
// 		changed = true;
// 	}
// 	p = p_sum;
// 	center_of_mass[0] = new_xm;
// 	center_of_mass[1] = new_ym;
// 	return changed;
// }

// void Node::recompute_center_of_mass(const dtype*x, const dtype*y)
// {//changes are propagated up the tree.
// 	bool changed = compute_center_of_mass(x,y);
// 	// int null_parent = parent == NULL;
// 	// fprintf(stdout,"\n\n\nCalled by level %d, condition %d, null parent %d\n\n\n",level,changed,null_parent);
// 	if(changed)
// 	{
// 		if(parent != NULL)
// 		{
// 			parent->recompute_center_of_mass(x,y);
// 		}
// 	}
// }

// Node* Node::remove_empty_nodes(int caller_id=-1)
// {//Returns the parent, if it needs to be deleted as well.
// 	//caller_id identifies the child that identified this node to be deleted.
// 	//the child with caller_id must be deleted as well.
// 	if( caller_id >= 0 )
// 	{
// 		delete child[caller_id];
// 		child[caller_id] = NULL;
// 	}
// 	if( p == 0 )
// 	{
// 		if(parent->getP() == 0)
// 		{
// 			return parent;
// 		}
// 	}
// 	return NULL;
// }

// Node* Node::get_extreme_child(int direction)
// {
// 	for(int i=0;i<4;++i)
// 	{
// 		int ii = ( direction == 0 ) ? i : 3-i;
// 		if( child[ii] != NULL )
// 		{
// 			return child[ii];
// 		}
// 	}
// 	return NULL;
// }



// Node* get_extreme_node(Node* root,int direction)
// {//direction: 0 for left, >0 for right
// 	Node* next = root;
// 	while(next != NULL)
// 	{
// 		Node* check = next->get_extreme_child(direction);
// 		if(check == NULL)
// 		{
// 			return next;
// 		}
// 		else
// 		{
// 			next = check;
// 		}
// 	}
// 	return NULL;
// }



// void remove_extreme_point_and_readjust(Node* root, const int direction, 
// 	const dtype*x, const dtype*y)
// {//Called on the node with the leaf to remove.
// 	//direction: 0 for left, >0 for right.
// 	Node* n = get_extreme_node(root, direction);
// 	n->remove_extreme_point(direction);
// 	n->recompute_center_of_mass(x,y);
// 	if( n->getP() == 0 )
// 	{
// 		Node* next = n->getParent();
// 		int level_index = n->getLevelIndex();
// 		while(next != NULL)
// 		{
// 			next = next->remove_empty_nodes(level_index);
// 		}
// 	}
// }

