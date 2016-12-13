#pragma once


// This segment represents the city and segment that are connected
//  and in the same direction, relative to the holder of the
//  Neighbor object.


struct Segment;

struct Neighbor
{
    int city; // city
    Segment* segment; // segment
};


