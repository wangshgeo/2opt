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

  // In this tree construction code we assume the points at within each node 
  // have the same Morton Key prefix. So when a point is on the exact boundary, 
  // it gets put into the quadrant it would go in if it was a small increment 
  // larger (in either x or y). This is fine, EXCEPT at the root. There is no 
  // next quadrant. So, we have to apply fudge factor to ranges to completely 
  // capture boundary points.
  const double fudge_factor = 1.00001;
  x_range *= fudge_factor;
  y_range *= fudge_factor;

  // cout << "x and y range: " << x_range << " " << y_range << endl;
  // cout << "x and y min: " << x_min << " " << y_min << endl;
  // cout << "x and y max: " << x_max << " " << y_max << endl;

  // The sorted morton key array is only needed for construction. After that, 
  // it is no longer needed.
  // So we can sort by the first item in pair, and retain the city index,
  // we create the following vector.
  vector< pair<morton_key_type, int> > morton_key_pairs; 
  point_morton_keys_ = new morton_key_type[tour.cities()];
  for(int i = 0; i < tour.cities(); ++i)
  {
    double x_normalized = ( tour.x(i) - x_min ) / x_range;
    double y_normalized = ( tour.y(i) - y_min ) / y_range;
    
    // filter normalized values
    x_normalized = (x_normalized < 0.0) ? 0.0: x_normalized;
    y_normalized = (y_normalized < 0.0) ? 0.0: y_normalized;
    x_normalized = (x_normalized > 1.0) ? 1.0: x_normalized;
    y_normalized = (y_normalized > 1.0) ? 1.0: y_normalized;

    MortonKey morton_key(i, x_normalized, y_normalized);
    // bitset<8*sizeof(morton_key.value())> morton_bits(morton_key.value());
    // cout << morton_bits.to_string().substr(22) << endl;
    pair<morton_key_type, int> morton_key_pair ( morton_key.value(), i );
    point_morton_keys_[i] = morton_key.value();
    morton_key_pairs.push_back( morton_key_pair );
  }
  std::sort(morton_key_pairs.begin(), morton_key_pairs.end());


  // Now we can create the tree (recursively)
  root_ = new QuadtreeNode( 
    nullptr, // parent 
    -1, // quadrant
    &morton_key_pairs[0], // Morton key pairs (key, point id)
    tour.cities(), // number of points / morton key pairs
    tour
  );

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