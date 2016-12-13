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
#include "Neighbour.h"

struct Tour
{
    Tour(const std::size_t cities, const DistanceTable& d);

    // Most useful member functions.
    template <std::size_t K>
    inline void exchange(SwapSet<K>&, const SwapMap<K>&);
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
    inline int getCityId(const SwapSet<K>& set, const int relIndex) const;
};


template <std::size_t K>
void Tour::exchange(SwapSet<K>& set, const SwapMap<K>& map)
{
    // The two points must be from different segments.
    for (std::size_t i = 0; i < K; ++i)
    {
        const int relativeCityA = map[2 * i];
        const int relativeCityB = map[2 * i + 1];
        const int segmentA = relativeCityA / 2;
        const int segmentB = relativeCityB / 2;
        const int pairIndexA = relativeCityA % 2;
        const int pairIndexB = relativeCityB % 2;
    }
    std::array<bool, K> modified;
    std::fill(modified.begin(), modified.end(), false);
    for (std::size_t i = 0; i < K; ++i)
    {
        const int j = map[2 * i];
        const int k = map[2 * i + 1];
        const int sj = j / 2;
        const int sk = k / 2;
        if (not modified[sj])
        {
            getNeighbourRef(set, sj) = getAsNeighbor(set, sk);
            modified[sj] = true;
        }
        else if (not modified[sk])
        {
            getNeighbourRef(set, sk) = getAsNeighbor(set, sj);
            modified[sk] = true;
        }
        else
        {
            std::cout << "Error! Swap failed.\n";
        }
    }
    // .
    SwapMap<K> cityIds;
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

template <std::size_t K>
void replace(SwapSet<K>& set, const int oldRelCity, const int newRelCity)
{
    getNeighbourRef(set, oldRelCity) = getAsNeighbor(set, newRelCity);
    getNeighbourRef(set, newRelCity) = getAsNeighbor(set, oldRelCity);
}


Neighbor getAsNeighbor(const SwapSet<K>& set, const int s, const int p) const
{
    return Neighbor{set[s]->n[p].c, set[s]};
}


Neighbor& getNeighborRef(SwapSet<K>& set, const int relCity)
{
    const int s = relCity / 2;
    const int p = relCity % 2;
    return set[s]->n[p];
}
Neighbor getAsNeighbor(const SwapSet<K>& set, const int relIndex) const
{
    const int s = relIndex / 2;
    const int p = relIndex % 2;
    return Neighbor{set[s]->n[p].c, set[s]};
}


int Tour::getCityId(const SwapSet<K>& set, const int relIndex) const
{
    const int segmentIndex = relIndex / 2;
    const int pairIndex = relIndex % 2;
    return set[segmentIndex].c[pairIndex];
}


int Tour::nextCity(const Segment* currentSegment, const int prevCity) const
{
    return (currentSegment->c[0] != prevCity)
        ? currentSegment->c[0] : currentSegment->c[1];
}
