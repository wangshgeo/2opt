#ifndef MORTON_KEY_H_
#define MORTON_KEY_H_

typedef unsigned int binary_type; // type of the binary representation of each 
  // coordinate.
typedef unsigned long int morton_key_type; // consists of interleaved 
  // binary_type
const int MAX_LEVEL = 21; //maximum level / depth of the quadtree. Leave at 
  // least one bit for flags.
const binary_type BINARY_TYPE_MAXIMUM = 1 << MAX_LEVEL; // to be multiplied by 
  // the normalized (0,1) coordinate.

// Deals with coordinates normalized to be in [0,1].
// 
class MortonKey
{
public:
  MortonKey(int city__, 
    double normalized_coordinate1, double normalized_coordinate2) : 
      city_(city__)
  {
    binary_type binary_coordinate1 = 
      static_cast<binary_type>( BINARY_TYPE_MAXIMUM * normalized_coordinate1 );
    binary_type binary_coordinate2 = 
      static_cast<binary_type>( BINARY_TYPE_MAXIMUM * normalized_coordinate2 );
    Interleave( binary_coordinate1, binary_coordinate2 );
  }
  int city() { return city_; }
  morton_key_type value() { return value_; }
private:
  morton_key_type value_; // the morton key value.
  int city_; // the city to which this morton key corresponds.
  void Interleave(
    binary_type binary_coordinate1, binary_type binary_coordinate2);
};



#endif 