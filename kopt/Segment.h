#pragma once


// Each segment consists of 2 city IDs, and 2 pointers
//  to neighbouring segments.


#include <array>

#include "Neighbor.h"

struct Segment
{
    std::array<Neighbor, 2> neighbors;
    int length = 0;

    inline Segment* nextSegment(const int nextCity) const;
    inline bool hasCity(const int city) const;
    inline int next(const int prev) const;
    inline Segment* next(const Segment* prev) const;
    Neighbor getNeighbor(const int pairIndex) const
};


Neighbor Segment::getNeighbor(const int pairIndex) const
{
    return neighbors[pairIndex];
}


void Segment::setNeighbor(Neighbor neighbor)
{
    if (neighbors[0].segment == nullptr)
    {
        neighbors[0] = neighbor;
    }
    else if (neighbors[1].segment == nullptr)
    {
        neighbors[1] = neighbor;
    }
}


void Segment::unsetNeighbor(const int city)
{
    if (neighbors[0].city == city)
    {
        neighbors[0].city = -1;
        neighbors[0].segment = nullptr;
    }
    else if (neighbors[1].city == city)
    {
        neighbors[1].city = -1;
        neighbors[1].segment = nullptr;
    }
}


void Segment::setNeighbor(Neighbor neighbor, const int pairIndex)
{
    neighbors[pairIndex] = neighbor;
    neighbors[pairIndex]->setNeighbor(neighbor);
}


int Segment::next(const int prev) const
{
    return (c[0] != prevCity) ? c[0] : c[1];
}


Segment* Segment::next(const Segment* prev) const
{
    return (s[0] != prev) ? s[0] : s[1];
}


Segment* Segment::nextSegment(const int nextCity) const
{
    return (s[0]->hasCity(nextCity)) ? s[0] : s[1];
}


bool Segment::hasCity(const int city) const
{
    return city == c[0] or city == c[1];
}

void Segment::replace(const int oldCity, const Neighbour& newNeighbour)
{
    if (n[0].c == oldCity)
    {
        n[0] = newNeighbour;
    }
    else
    {
        n[1] = newNeighbour;
    }
}


Segment* Segment::follow(const int city)
{
    if (s[0]->hasCity(city))
    {
        return s[0];
    }
    else if (s[1]->hasCity(city))
    {
        return s[1];
    }
    return nullptr;
}


