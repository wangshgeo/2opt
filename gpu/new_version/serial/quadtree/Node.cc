#include "Node.h"

void Node::deleteSegment(int segment)
{ 
	for(int i=0;i<s;++i)
	{
		if(segments[i] == segment) segments.erase( segments.begin() + i );
	}
}

void Node::remove_extreme_point(int direction)
{//direction: 0 for left, >0 for right.
	if(direction == 0)
	{
		points.erase(points.begin());
	}
	else
	{
		points.pop_back();
	}
}

bool Node::compute_center_of_mass(const dtype*x,const dtype*y)
{//Returns true if center of mass changed, false if not.
	bool changed = false;
	dtype xm_sum = 0;
	dtype ym_sum = 0;
	int p_sum = 0;
	for(int i=0;i<4;++i)
	{
		Node* c = child[i];
		if(c != NULL)
		{
			int cp = c->getP();
			p_sum += cp;
			dtype* com = c->getCenterOfMass();
			xm_sum += cp * com[0];
			ym_sum += cp * com[1];
		}
	}
	for(unsigned int i=0;i<points.size();++i)
	{
		++p_sum;
		xm_sum += x[points[i]];
		ym_sum += y[points[i]];
	}
	dtype new_xm = (p_sum > 0) ? (xm_sum / p_sum) : 0;
	dtype new_ym = (p_sum > 0) ? (ym_sum / p_sum) : 0;
	if ( (p != p_sum) or (new_xm != center_of_mass[0]) or (new_ym != center_of_mass[1]) )
	{
		changed = true;
	}
	p = p_sum;
	center_of_mass[0] = new_xm;
	center_of_mass[1] = new_ym;
	return changed;
}

void Node::recompute_center_of_mass(const dtype*x, const dtype*y)
{//changes are propagated up the tree.
	bool changed = compute_center_of_mass(x,y);
	// int null_parent = parent == NULL;
	// fprintf(stdout,"\n\n\nCalled by level %d, condition %d, null parent %d\n\n\n",level,changed,null_parent);
	if(changed)
	{
		if(parent != NULL)
		{
			parent->recompute_center_of_mass(x,y);
		}
	}
}

Node* Node::remove_empty_nodes(int caller_id=-1)
{//Returns the parent, if it needs to be deleted as well.
	//caller_id identifies the child that identified this node to be deleted.
	//the child with caller_id must be deleted as well.
	if( caller_id >= 0 )
	{
		delete child[caller_id];
		child[caller_id] = NULL;
	}
	if( p == 0 )
	{
		if(parent->getP() == 0)
		{
			return parent;
		}
	}
	return NULL;
}

Node* Node::get_extreme_child(int direction)
{
	for(int i=0;i<4;++i)
	{
		int ii = ( direction == 0 ) ? i : 3-i;
		if( child[ii] != NULL )
		{
			return child[ii];
		}
	}
	return NULL;
}



Node* get_extreme_node(Node* root,int direction)
{//direction: 0 for left, >0 for right
	Node* next = root;
	while(next != NULL)
	{
		Node* check = next->get_extreme_child(direction);
		if(check == NULL)
		{
			return next;
		}
		else
		{
			next = check;
		}
	}
	return NULL;
}



void remove_extreme_point_and_readjust(Node* root, const int direction, 
	const dtype*x, const dtype*y)
{//Called on the node with the leaf to remove.
	//direction: 0 for left, >0 for right.
	Node* n = get_extreme_node(root, direction);
	n->remove_extreme_point(direction);
	n->recompute_center_of_mass(x,y);
	if( n->getP() == 0 )
	{
		Node* next = n->getParent();
		int level_index = n->getLevelIndex();
		while(next != NULL)
		{
			next = next->remove_empty_nodes(level_index);
		}
	}
}

