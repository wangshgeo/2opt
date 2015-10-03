#ifndef SEGMENT_H_
#define SEGMENT_H_

class QuadtreeNode;

typedef long int cost_t;

// This represents a path segment between two adjacent cities.
// A tour is made of a number of segments the same as the number of cities.
class Segment
{
public:
  int start_city; // index of the starting city of this segment.
  int end_city; // index of the ending city of this segment.
  int order; // The order of this segment in the global tour.
  cost_t length; // length (or cost) of this segment.
  // The coordinates of the bisecting point.
  double center_x;
  double center_y;
  // The quadtree node at which the segment resides.
  QuadtreeNode *node;
  Segment();
};

#endif