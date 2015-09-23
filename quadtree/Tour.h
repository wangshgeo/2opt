#ifndef TOUR_H_
#define TOUR_H_

#include <cmath>

#include "Instance.h"
#include "Segment.h"

// From an Instance, generates a Tour to be used for computations.
// Assumes the Instance destructor takes care of the data structures derived 
// from it.
// Segments are indexed by the city from which it originates.
class Tour
{
public:
  Tour(Instance& instance)
  {
    cities_ = instance.cities();
    x_ = instance.x();
    y_ = instance.y();
    segments_ = new Segment[cities_];
    InitializeSegments();
  }
  int cities() { return cities_; }
  double* x() { return x_; }
  double* y() { return y_; }
  double x(int i) { return x_[i]; }
  double y(int i) { return y_[i]; }
  Segment* segment(int index) { return &segments_[index]; }
  void Swap(Segment& first_deleted, Segment& second_deleted);
  ~Tour()
  {
    delete[] segments_;
  }
  void SerialCheck();
  void Check();
private:
  int cities_; // number of cities.
  double *x_,*y_; // city coordinates, x[city_index], y[city_index]
  Segment* segments_; // the tour path is represented as a series of 
    // ordered segments.
  void InitializeSegments();
  void reverse_segment(Segment& interior, 
    Segment& first_deleted, Segment& second_deleted);
  void renew_segment(Segment& segment, 
    int new_start_city, int new_end_city);
  cost_t Cost(int city1, int city2);
  bool adjacent_segments(Segment& s1, Segment& s2);
  cost_t SwapCost(Segment& s1, Segment& s2);
};


#endif