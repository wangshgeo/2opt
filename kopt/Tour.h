#pragma once

// A tour consists of a sequence of segments.
// City ids range from [0, m_s.size())

#include <algorithm>
#include <vector>

#include "Segment.h"


struct Tour
{
    Tour(const std::size_t cities, const DistanceTable& d);

    template <std::size_t K>
    inline void exchange(const Segment* s[K], const SwapSet<K>&);

    int length() const;
    bool valid() const;

    std::vector<Segment> m_s;

private:
    void initialize(const std::size_t cities);
    void connect();
    void computeDistances(const DistanceTable& d);
};


template <std::size_t K>
void Tour::exchange(
    const Segment* segments[K], const int swapSet[2 * K]);
{
    for (std::size_t i = 0; i < K; ++i)
    {
        int segmentIndex = swapSet[2 * K] / 2;
        int pairIndex = swapSet[2 * K] % 2;
        
        segments[segmentIndex];
    }
        swapSet
        segments[]
    std::reverse(m_tour.begin() + low + 1, m_tour.begin() + high + 1);
}



