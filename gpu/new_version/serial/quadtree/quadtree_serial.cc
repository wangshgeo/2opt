#include "quadtree_serial.h"

struct pair_search_comparator
{
    bool operator()(morton_key_type pair, size_t size) const
    {
    	return pair.first < size;
    }
    bool operator()(size_t size, morton_key_type pair) const
    {
    	return size < pair.first;
    }
};



void point_metrics(const morton_key_type* morton_keys, const int size,
	const dtype* x, const dtype* y,
	dtype* rmax, dtype* xavg, dtype* yavg)
{
	dtype xsum = 0;
	dtype ysum = 0;
	for(int i=0;i<size;++i)
	{
		xsum += x[morton_keys[i].second];
		ysum += y[morton_keys[i].second];
	}
	*xavg = xsum / size;
	*yavg = ysum / size;
	*rmax = 0;
	for(int i=0;i<size;++i)
	{
		dtype dx = x[morton_keys[i].second] - *xavg;
		dtype dy = y[morton_keys[i].second] - *yavg;
		dtype r = sqrt(dx*dx + dy*dy);
		*rmax = (r < *rmax) ? *rmax: r;
	}
}



Node* construct_quadtree_serial(Node* parent, int level_index, 
	const morton_key_type* morton_keys, const int size, 
	const int current_level, const dtype*x, const dtype*y)
{
	Node* current_node = new Node( parent, current_level, level_index );
	dtype rmax,xavg,yavg;
	point_metrics(morton_keys, size, x, y, &rmax, &xavg, &yavg);
	current_node->setDiameter(rmax);
	current_node->setCenterOfMass(xavg,yavg);
	current_node->setP(size);

	bool small = size <= 2;
	bool same_morton_keys = morton_keys[0].second==morton_keys[size-1].second;
	bool maximum_level = current_level == MAX_LEVEL;
	if( small or same_morton_keys or maximum_level )
	{
		for(int i=0;i<size;++i)
		{
			int index = morton_keys[i].second;
			current_node->addPoint(index);
		}
	}
	else
	{
		//quadrant keys.
		mtype all_ones = ~ ( (mtype) 0 );
		// std::bitset<sizeof(mtype)*8> aob(all_ones);
		// std::cout << "\tall_ones: " << aob << std::endl;
		int suffix_bits = 2*(MAX_LEVEL - current_level);
		mtype prefix_mask = all_ones << suffix_bits;
		// std::bitset<sizeof(mtype)*8> pmb(prefix_mask);
		// std::cout << "\tprefix_mask: " << pmb << std::endl;
		mtype prefix = morton_keys[0].first & prefix_mask;//obviously, we assume keys have the same prefix for current level.
		mtype q2_suffix =  ( (mtype) 1 ) << ( suffix_bits - 2 );
		// std::bitset<sizeof(mtype)*8> q2sb(q2_suffix);
		// std::cout << "\tq2_suffix: " << q2sb << std::endl;
		mtype q3_suffix =  ( (mtype) 2 ) << ( suffix_bits - 2 );
		mtype q4_suffix =  ( (mtype) 3 ) << ( suffix_bits - 2 );

		mtype q2 = prefix + q2_suffix;
		mtype q3 = prefix + q3_suffix;
		mtype q4 = prefix + q4_suffix;

		// std::bitset<sizeof(mtype)*8> q2b(q2);
		// std::cout << "\tq2: " << q2b << std::endl;
		// std::bitset<sizeof(mtype)*8> q3b(q3);
		// std::cout << "\tq3: " << q3b << std::endl;
		// std::bitset<sizeof(mtype)*8> q4b(q4);
		// std::cout << "\tq4: " << q4b << std::endl;

		const morton_key_type* q2_start = std::lower_bound(morton_keys, morton_keys+size, q2, pair_search_comparator());
		int size_q1 = q2_start-morton_keys;
		if(size_q1 > 0)
		{
			current_node->setChild(
				construct_quadtree_serial(current_node, 0, morton_keys,size_q1,current_level+1, x,y),
				0 );
			// fprintf(stdout, "Made a node at level %d\n",current_level+1);
		}

		const morton_key_type* q3_start = std::lower_bound(q2_start, morton_keys+size, q3, pair_search_comparator());
		int size_q2 = q3_start-q2_start;
		if(size_q2 > 0)
		{
			current_node->setChild( 
				construct_quadtree_serial(current_node, 1, q2_start,size_q2,current_level+1, x,y),
				1 );
			// fprintf(stdout, "Made a node at level %d\n",current_level+1);
		}

		const morton_key_type* q4_start = std::lower_bound(q3_start, morton_keys+size, q4, pair_search_comparator());
		int size_q3 = q4_start-q3_start;
		if(size_q3 > 0)
		{
			current_node->setChild( 
				construct_quadtree_serial(current_node, 2, q3_start,size_q3,current_level+1, x,y),
				2 );
			// fprintf(stdout, "Made a node at level %d\n",current_level+1);
		}

		int size_q4 = (morton_keys + size) - q4_start;
		if(size_q4 > 0)
		{
			current_node->setChild( 
				construct_quadtree_serial(current_node, 3, q4_start,size_q4,current_level+1, x,y),
				3 );
			// fprintf(stdout, "Made a node at level %d\n",current_level+1);
		}
	}
	return current_node;
}

void destroy_quadtree_serial(Node* current_node)
{
	for(int i=0;i<4;++i)
	{
		Node* child = current_node->getChild(i);
		if(child != NULL)
		{
			// fprintf(stdout, "destroying child %d on level %d\n",i,current_node->getLevel());
			destroy_quadtree_serial(child);
		}
	}
	delete current_node;
}


static bool far_enough(const dtype* segment_center, const dtype* center_of_mass, 
	const cost_t diameter_sum)
{	
	dtype dx = segment_center[0] - center_of_mass[0];
	if(dx > diameter_sum) return true;
	dtype dy = segment_center[1] - center_of_mass[1];
	if(dy > diameter_sum) return true;
	dtype ds = sqrt(dx*dx + dy*dy);
	if(ds > diameter_sum) 
		return true;
	else 
		return false;
}

static bool compare_pair( int* i_best, int* j_best, cost_t *cost_best,
	const int i, const int j,
	const cost_t* segment_lengths, const dtype* x, const dtype* y, const int n)
{
	//Returns true if better swap found, false otherwise.
	cost_t old_cost = segment_lengths[i] + segment_lengths[j];
	dtype dx = abs(x[i] - x[j]);
	dtype dy = abs(y[i] - y[j]);
	if(dx > old_cost or dy > old_cost) return false;
	dtype ds1 = sqrt(dx*dx + dy*dy);
	if(ds1 > old_cost) return false;
	int i_next = i + 1;
	i_next = (i_next == n) ? 0: i_next;
	int j_next = j + 1;
	j_next = (j_next == n) ? 0: j_next;
	dx = abs(x[i_next] - x[j_next]);
	dy = abs(y[i_next] - y[j_next]);
	if(ds1 + dx > old_cost or ds1 + dy > old_cost) return false;
	dtype ds2 = sqrt(dx*dx + dy*dy);
	if(ds1 + ds2 > old_cost)
	{
		return false;
	} 
	else
	{
		*i_best = i;
		*j_best = j;
		*cost_best = (cost_t) (ds1 + ds2);
		return true;
	}
}

static bool check_node_segments( int* i_best, int* j_best, cost_t *cost_best,
	int* best_segment_index,
	const int segment_index, const cost_t* segment_lengths,
	const leaf_container* segments,
	const dtype* x, const dtype* y, const int n, const int* map)
{
	//Returns true if better swap found, otherwise false.
	bool swap = false;
	int i = map[segment_index];
	for( int segment=0; segment < (int)segments->size(); ++segment )
	{
		int j = map[(*segments)[segment]];
		bool swap_ = compare_pair(i_best, j_best, cost_best,
			i, j, segment_lengths, x, y, n);
		if(swap_)
		{
			swap = true;
			*best_segment_index = segment;
		}
	}
	return swap;
}

static void traverse_quadtree( int* i_best, int* j_best, cost_t *cost_best,
	Node** best_node, int* best_segment_index,
	const int segment_index, const dtype* segment_center, Node* tree,
	const cost_t* segment_lengths, const dtype* x, const dtype* y, const int n, 
	const int* map)
{
	bool skip = far_enough( segment_center, tree->getCenterOfMass(), 
		tree->getDiameter() + segment_lengths[segment_index] );
	if( not skip )
	{
		bool swap = check_node_segments(i_best, j_best, cost_best, best_segment_index, 
			segment_index, segment_lengths, tree->getSegments(), x, y, n, map);
		if(swap) *best_node = tree;
		for(int i=0;i<4;++i)
		{
			if(tree->getChild(i) != NULL)
			{
				traverse_quadtree( i_best, j_best, cost_best, best_node, best_segment_index,
					segment_index, segment_center, tree->getChild(i), 
					segment_lengths, x, y, n, map );
			}
		}
	}
}

void best_improvement_quadtree( int* i_best, int* j_best, cost_t *cost_best,
	Node** best_node, int* best_segment_index,
	const cost_t* segment_lengths, const dtype* center_x, const dtype* center_y,
	const dtype *x, const dtype *y,	const int n, 
	Node* tree,	int*map )
{
	for(int segment_index = 0; segment_index < n; ++segment_index)
	{
		dtype segment_center[2];
		segment_center[0] = center_x[segment_index];
		segment_center[1] = center_y[segment_index];
		traverse_quadtree( i_best, j_best, cost_best, best_node, best_segment_index,
			segment_index, segment_center, tree,
			segment_lengths, x, y, n, map );
	}

}