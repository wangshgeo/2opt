#include "segment_serial.h"

// Computes and writes out the segment lengths of n-length tour represented 
// by x and y to segments.
void compute_segment_lengths(const dtype* x, const dtype* y, const int n, 
	cost_t* segments)
{
	for(int i=0;i<n-1;++i)
	{
		dtype dx = x[i+1] - x[i];
		dtype dy = y[i+1] - y[i];
		segments[i] = ( (cost_t) sqrt(dx*dx + dy*dy) );
	}
	dtype dx = x[0] - x[n-1];
	dtype dy = y[0] - y[n-1];
	segments[n-1] = ( (cost_t) sqrt(dx*dx + dy*dy) );
}

void ordered_point_morton_keys( const morton_key_type* morton_pairs, 
	mtype* morton_keys, const int n )
{
	for(int i=0;i<n;++i)
	{
		morton_keys[morton_pairs[i].second] = morton_pairs[i].first;
	}
}

// Returns the two bits at a certain level in a morton key.
mtype get_level_msb( const mtype key, const int level)
{
	int shift = 2*( MAX_LEVEL - level );
	mtype mask = (mtype) 3;
	mtype shifted = key >> shift;
	return shifted & mask;
}

void insert_segment( Node* tree, const mtype* point_morton_keys, 
	const int segment_index, const int next_segment_index )
{
	Node* current_node = tree;
	tree->increment_total_s();
	int next_level = 1;
	int level_index0 = get_level_msb(point_morton_keys[segment_index],next_level);
	int level_index1 = get_level_msb(point_morton_keys[next_segment_index],next_level);
	// fprintf(stdout, "Segment %d (%d %d):\n",segment_index,level_index0,level_index1);
	// std::cout << std::bitset<sizeof(mtype)*8>(point_morton_keys[segment_index]) << std::endl;
	// std::cout << std::bitset<sizeof(mtype)*8>(point_morton_keys[next_segment_index]) << std::endl;

	// std::cout << std::bitset<sizeof(mtype)*8>(level_index0) << std::endl;
	// std::cout << std::bitset<sizeof(mtype)*8>(level_index1) << std::endl;
	// std::cout << "Descending: \n";
	while( level_index0 == level_index1 )
	{
		Node* current_child = current_node->getChild(level_index0);
		if( current_child == NULL )	break;
		// std::cout << "\tCurrent level: " << next_level << " (quadrant: " << level_index0 << ")\n";
		current_node = current_child;
		current_node->increment_total_s();
		++next_level;
		level_index0 = get_level_msb(point_morton_keys[segment_index],next_level);
		// std::cout << std::bitset<sizeof(mtype)*8>(level_index0) << std::endl;
		level_index1 = get_level_msb(point_morton_keys[next_segment_index],next_level);
		// std::cout << std::bitset<sizeof(mtype)*8>(level_index1) << std::endl;
	}
	current_node->addSegment(segment_index);
}


void insert_segments( Node* tree, const mtype* point_morton_keys, const int n)
{
	for(int i=0;i<n-1;++i)
	{
		insert_segment(tree, point_morton_keys, i, i+1);
	}
	insert_segment(tree, point_morton_keys, n-1, 0);
}


void delete_segment( Node* tree, const mtype* point_morton_keys, 
	const int segment_index, const int next_segment_index )
{
	Node* current_node = tree;
	int current_level = 1;
	int level_index0 = get_level_msb(point_morton_keys[segment_index],current_level);
	int level_index1 = get_level_msb(point_morton_keys[next_segment_index],current_level);
	while( level_index0 == level_index1 )
	{
		current_node = current_node->getChild(level_index0);
		++current_level;
		level_index0 = get_level_msb(point_morton_keys[segment_index],current_level);
		level_index1 = get_level_msb(point_morton_keys[next_segment_index],current_level);
	}
	current_node->deleteSegment(segment_index);
}

void compute_segment_centers(dtype* segment_center_x, dtype* segment_center_y, 
	const dtype* x, const dtype* y, const int n)
{
	for( int i = 0; i < n - 1; ++i )
	{
		segment_center_x[i] = (x[i+1] - x[i]) / 2;
		segment_center_y[i] = (y[i+1] - y[i]) / 2;
	}
	segment_center_x[n] = (x[0] - x[n-1]) / 2;
	segment_center_y[n] = (y[0] - y[n-1]) / 2;
}