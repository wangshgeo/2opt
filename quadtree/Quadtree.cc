#include "Quadtree.h"

using namespace std;

Quadtree::Quadtree(Tour& tour)
{
  // Get domain bounds
  double x_min = minimum(tour.x(), tour.cities());
  double x_max = maximum(tour.x(), tour.cities());
  double y_min = minimum(tour.y(), tour.cities());
  double y_max = maximum(tour.y(), tour.cities());

  double x_range = x_max - x_min;
  double y_range = y_max - y_min;

  // The sorted morton key array is only needed for construction. After that, 
  // it is no longer needed.
  // So we can sort by the first item in pair, and retain the city index,
  // we create the following vector.
  vector< pair<morton_key_type, int> > morton_key_pairs; 
  for(int i = 0; i < tour.cities(); ++i)
  {
    double x_normalized = ( tour.x(i) - x_min ) / x_range;
    double y_normalized = ( tour.y(i) - y_min ) / y_range;
    
    // filter normalized values
    x_normalized = (x_normalized <= 0.0) ? 0.0: x_normalized;
    y_normalized = (y_normalized <= 0.0) ? 0.0: y_normalized;
    x_normalized = (x_normalized >= 1.0) ? 1.0: x_normalized;
    y_normalized = (y_normalized >= 1.0) ? 1.0: y_normalized;

    MortonKey morton_key(i, x_normalized, y_normalized);
    pair<morton_key_type, int> morton_key_pair( morton_key.value(), i );
    morton_key_pairs.push_back( morton_key_pair );
  }
  std::sort(morton_key_pairs.begin(), morton_key_pairs.end());

  // Now we can create the tree (recursively)
  

}


double Quadtree::minimum(double* x, int length)
{
  double minimum = x[0];
  for(int i = 0; i < length; ++i) minimum = (x[i] < minimum) ? x[i]: minimum;
  return minimum;
}
double Quadtree::maximum(double* x, int length)
{
  double maximum = x[0];
  for(int i = 0; i < length; ++i) maximum = (x[i] > maximum) ? x[i]: maximum;
  return maximum;
}