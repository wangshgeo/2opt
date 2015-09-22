#include "MortonKey.h"


void MortonKey::Interleave(
    binary_type binary_coordinate1, binary_type binary_coordinate2)
{
  value_ = static_cast<morton_key_type>( 0 );
  morton_key_type one = static_cast<morton_key_type>( 1 );
  int bytes = sizeof(binary_coordinate1);
  int bits = 8*bytes;
  for(int i = bits-1; i >= 0; --i)
  {
    value_ |= (binary_coordinate1 >> i) & one;
    value_ <<= 1;
    value_ |= (binary_coordinate2 >> i) & one;
    if(i != 0) value_ <<= 1;
  }
}