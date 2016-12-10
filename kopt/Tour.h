#pragma once


// A tour consists of a sequence of segments.
// City ids range from [0, m_s.size())


#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "DistanceTable.h"
#include "Segment.h"
#include "SwapMap.h"
#include "SwapSet.h"


struct Tour
{
    Tour(const std::size_t cities, const DistanceTable& d);

    // Most useful member functions.
    template <std::size_t K>
    inline void exchange(SwapSet&, const SwapMap&);
    int length() const;
    inline int nextCity(const Segment* currentSegment, const int prevCity) const;

    // For debugging purposes.
    bool valid() const;
    void print() const;

    // Member variables.
    std::vector<Segment> m_s;
    const DistanceTable& m_d;
private:
    void initialize(const std::size_t cities);
    void connect();
    void computeDistances();
};


template <std::size_t K>
void Tour::exchange(SwapSet& set, const SwapMap& map)
{
    // Prevent overwriting prematurely.
    SwapMap cityIds;
    for (std::size_t i = 0; i < K; ++i)
    {
        cityIds[i] = set[map[i]]->c;
    }
    // Assign new cities to segments and reconnect.
    for (std::size_t i = 0; i < K; ++i)
    {
        set[i]->c = cityIds[i];
    }
    // Recompute distances.
    for (std::size_t i = 0; i < K; ++i)
    {
        set[i]->length = m_d.get(i, set[i]->c);
        std::cout
            << "segment: " << i << ", "
            << set[i]->c << "\n";
    }
}

int Tour::nextCity(const Segment* currentSegment, const int prevCity) const
{
    return (currentSegment->c[0] != prevCity)
        ? currentSegment->c[0] : currentSegment->c[1];
}
