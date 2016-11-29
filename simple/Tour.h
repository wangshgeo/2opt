#pragma once

// A tour consists of a sequence of city identifiers.

#include <algorithm>
#include <vector>

#include "DistanceTable.h"

class Tour
{
    public:
        Tour(const int cityCount);
        Tour(std::vector<int> initial);

        inline int getCityCount() const;
        inline int getCityId(const int sequenceNumber) const;
        inline int getNextCityId(size_t sequenceNumber) const;
        bool valid() const;
        inline const std::vector<int>& getTour() const;

        inline void exchange(const int i, const int j);
        inline void shuffle();
        inline void perturb();
        int length(const DistanceTable& d) const;
     private:
        std::vector<int> m_tour;
};


void Tour::exchange(const int i, const int j)
{
    const bool iless = i < j;
    const int low   = (iless) ? i : j;
    const int high  = (iless) ? j : i;
    std::reverse(m_tour.begin() + low + 1, m_tour.begin() + high + 1);
}


int Tour::getCityCount() const
{
    return m_tour.size();
}


int Tour::getCityId(const int sequenceNumber) const
{
    return m_tour[sequenceNumber];
}


int Tour::getNextCityId(size_t sequenceNumber) const
{
    sequenceNumber = (sequenceNumber + 1 == m_tour.size())
        ? 0 : sequenceNumber + 1;
    return m_tour[sequenceNumber];
}


const std::vector<int>& Tour::getTour() const
{
    return m_tour;
}


void Tour::shuffle()
{
    std::random_shuffle(m_tour.begin(), m_tour.end());
}


void Tour::perturb()
{
    const int i = std::rand() % m_tour.size();
    const int j = std::rand() % m_tour.size();
    exchange(i, j);
}


