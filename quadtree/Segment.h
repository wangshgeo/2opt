#ifndef SEGMENT_H_
#define SEGMENT_H_

typedef long int cost_t;

// This represents a path segment between two adjacent cities.
// A tour is made of a number of segments the same as the number of cities.
typedef struct Segment
{
  int start_city; // index of the starting city of this segment.
  int end_city; // index of the ending city of this segment.
  int order; // The order of this segment in the global tour.
  cost_t length; // length (or cost) of this segment.
} Segment;

#endif