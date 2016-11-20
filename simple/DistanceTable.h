#pragma once

// Provides the symmetric distance between a pair of cities.

#include <cmath>

#include <vector>

#include "City.h"
#include "IndexHash.h"

class DistanceTable
{
    public:
        enum class DistanceFunction : char
        {
            EUC = 'E',
            GEO = 'G'
        };
        DistanceTable(const std::vector<City>& cities,
            const DistanceFunction f);
        inline int getDistance(const int i, const int j) const;
    private:
        std::vector<int> m_distances;
        const IndexHash m_hash;

        inline int round(const double x) const;
        inline int distance(const std::vector<City>& cities,
            const int i, const int j) const;
        inline double toRadians(const double coordinate) const;
        inline int geoDistance(const std::vector<City>& cities,
            const int i, const int j) const;
};


int DistanceTable::getDistance(const int i, const int j) const
{
  return m_distances[m_hash.hash(i, j)];
}

