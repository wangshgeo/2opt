#include "MortonKey.h"

static const morton_key_type MORTON_ALL_ZEROS
  = static_cast<morton_key_type>(0);
static const morton_key_type MORTON_ALL_ONES = ~MORTON_ALL_ZEROS;
static const morton_key_type MORTON_ONE = static_cast<morton_key_type>( 1 );
static const morton_key_type MORTON_TWO = static_cast<morton_key_type>( 2 );
static const morton_key_type MORTON_THREE = static_cast<morton_key_type>( 3 );

using namespace std;

void MortonKey::Interleave(
    binary_type binary_coordinate1, binary_type binary_coordinate2)
{
  value_ = MORTON_ALL_ZEROS;
  int bytes = sizeof(binary_coordinate1);
  int bits = 8*bytes;
  for(int i = bits-1; i >= 0; --i)
  {
    value_ |= (binary_coordinate1 >> i) & MORTON_ONE;
    value_ <<= 1;
    value_ |= (binary_coordinate2 >> i) & MORTON_ONE;
    if(i != 0) value_ <<= 1;
  }
}

vector<morton_key_type> ExtractLeadingQuadrants(
  morton_key_type node_morton_key, int tree_level)
{
  // Determine prefix mask.
  int suffix_bits = 2*(MAX_LEVEL - tree_level - 1); // we subtract one because 
    // the quadrants are one level down from the current tree_level_.
  morton_key_type prefix_mask = MORTON_ALL_ONES << suffix_bits;

  // Obviously, we assume keys have the same prefix for a given node.
  morton_key_type prefix = node_morton_key & prefix_mask;

  // now determine the quadrant morton number.
  morton_key_type suffix01 =  MORTON_ONE << ( suffix_bits - 2 );
  morton_key_type suffix02 =  MORTON_TWO << ( suffix_bits - 2 );
  morton_key_type suffix03 =  MORTON_THREE << ( suffix_bits - 2 );

  vector<morton_key_type> quadrant_keys;
  quadrant_keys.push_back(prefix);
  quadrant_keys.push_back(prefix+suffix01);
  quadrant_keys.push_back(prefix+suffix02);
  quadrant_keys.push_back(prefix+suffix03);
  
  return quadrant_keys;
}