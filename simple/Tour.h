#pragma once

// A tour consists of a sequence of city identifiers.

#include <algorithm>
#include <vector>


class Tour
{
    public:
        Tour(std::vector<int> initial);
        inline int getCityCount() const;
        inline int getCityId(const int sequenceNumber) const;
        inline int getNextCityId(int sequenceNumber) const;
        inline void exchange(const int i, const int j);
        bool valid() const;
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


int Tour::getNextCityId(int sequenceNumber) const
{
    sequenceNumber = (sequenceNumber == m_tour.size() - 1)
        ? -1 : sequenceNumber;
    return m_tour[sequenceNumber + 1];
}



