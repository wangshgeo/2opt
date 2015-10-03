#include "Segment.h"

Segment::Segment()
{
  start_city = -1; // index of the starting city of this segment.
  end_city = -1; // index of the ending city of this segment.
  order = -1; // The order of this segment in the global tour.
  length = 0; // length (or cost) of this segment.
  // The coordinates of the bisecting point.
  center_x = 0;
  center_y = 0;
  // The quadtree node at which the segment resides.
  node = nullptr;
}