#pragma once


// Each segment consists of two city IDs.
//  One is specified in the segment, and the other is implict
//  as the index of this segment in the segment container.


struct Segment
{
    int c = -1;  // one city in this segment.
    Segment* i = nullptr;  // connects to implied city ID.
    int length = 0;

    inline int nextCity(const int prevCity) const;
    inline Segment* nextSegment(const int nextCity) const;
    inline bool hasCity(const int city) const;
};


int Segment::nextCity(const int prevCity) const
{
    return (c[0] != prevCity) ? c[0] : c[1];
}


Segment* Segment::nextSegment(const int nextCity) const
{
    return (s[0]->hasCity(nextCity)) ? s[0] : s[1];
}


bool Segment::hasCity(const int city) const
{
    return city == c[0] or city == c[1];
}


