#pragma once

// A tour consists of a sequence of segments.
// City ids range from [0, m_s.size())


#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "Segment.h"
#include "DistanceTable.h"


struct Tour
{
    Tour(const std::size_t cities, const DistanceTable& d);

    template <std::size_t K>
    inline void exchange(std::array<Segment*, K>&, const std::array<int, 2 * K> swapSet);

    int length() const;
    bool valid() const;

    std::vector<Segment> m_s;
    const DistanceTable& m_d;
private:
    void initialize(const std::size_t cities);
    void connect();
    void computeDistances();
    template <std::size_t K>
    inline int getCityId(
        const std::array<Segment*, K>&, const int relIndex) const;
    template <std::size_t K>
    inline void setCityId(
        std::array<Segment*, K>&, const int relIndex, const int cityId);
};


template <std::size_t K>
void Tour::exchange(std::array<Segment*, K>& segments, const std::array<int, 2 * K> swapSet)
{
    std::array<int, 2 * K> relativeMap;
    std::array<int, 2 * K> cities;
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
    for (auto& s : segments)
    {
        s->length = m_d.get(s->c[0], s->c[1]);
        std::cout << "segment: " << s->c[0] << ", " << s->c[1] << "\n";
    }
}

template <std::size_t K>
int Tour::getCityId(
        const std::array<Segment*, K>& segments, const int relIndex) const
{
    int segmentIndex = relIndex / 2;
    int pairIndex = relIndex % 2;
    return segments[segmentIndex]->c[pairIndex];
}


template <std::size_t K>
void Tour::setCityId(
        std::array<Segment*, K>& segments, const int relIndex, const int cityId)
{
    int segmentIndex = relIndex / 2;
    int pairIndex = relIndex % 2;
    segments[segmentIndex]->c[pairIndex] = cityId;
}


