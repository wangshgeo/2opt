#pragma once

// Provides the symmetric distance between a pair of cities.

#include <cmath>

#include <vector>

#include "City.h"
#include "CostFunction.h"
#include "IndexHash.h"

class DistanceTable
{
    public:
        DistanceTable(const std::vector<City>& cities,
            const CostFunction f);
        inline int get(const int i, const int j) const;
    private:
        std::vector<int> m_distances;
        const IndexHash m_hash;

        inline int nearestInt(const double x) const;
        inline int distance(const std::vector<City>& cities,
            const int i, const int j) const;
        inline double toGeographic(const double coordinate) const;
        inline int geoDistance(const std::vector<City>& cities,
            const int i, const int j) const;
};


int DistanceTable::getDistance(const int i, const int j) const
{
  return m_distances[m_hash.hash(i, j)];
}


