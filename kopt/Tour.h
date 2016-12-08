#pragma once

// A tour consists of a sequence of segments.
// City ids range from [0, m_s.size())

#include <algorithm>
#include <vector>

#include "Segment.h"
#include "DistanceTable.h"


struct Tour
{
    Tour(const std::size_t cities, const DistanceTable& d);

    template <std::size_t K>
    inline void exchange(Segment* s[K], const int swapSet[2 * K]);

    int length() const;
    bool valid() const;

    std::vector<Segment> m_s;

private:
    void initialize(const std::size_t cities);
    void connect();
    void computeDistances(const DistanceTable& d);
    inline int getCityId(
        Segment** segments, const int relIndex) const;
    inline void setCityId(
        Segment** segments, const int relIndex, const int cityId);
};


template <std::size_t K>
void Tour::exchange(
    Segment* segments[K], const int swapSet[2 * K])
{
    int relativeMap[2 * K];
    int cities[2 * K];
    for (std::size_t i = 0; i < K; ++i)
    {
        int j = swapSet[2 * i];
        int k = swapSet[2 * i + 1];
        relativeMap[j] = k;
        relativeMap[k] = j;
        cities[j] = getCityId(segments, j);
        cities[k] = getCityId(segments, k);
    }
    for (std::size_t i = 0; i < 2 * K; ++i)
    {
        setCityId(segments, i, cities[relativeMap[i]]);
    }
}


int Tour::getCityId(
    Segment** segments, const int relIndex) const
{
    int segmentIndex = relIndex / 2;
    int pairIndex = relIndex % 2;
    return segments[segmentIndex]->c[pairIndex];
}


void Tour::setCityId(
    Segment** segments, const int relIndex, const int cityId)
{
    int segmentIndex = relIndex / 2;
    int pairIndex = relIndex % 2;
    segments[segmentIndex]->c[pairIndex] = cityId;
}


