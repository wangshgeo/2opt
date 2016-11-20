#pragma once

#include <cmath>

#include <algorithm>
#include <vector>

#include "City.h"
#include "DistanceTable.h"
#include "IndexHash.h"


class World
{
  public:
    World() = default;
    World(std::vector<City> cities);
    inline double getDistance(const int i, const int j) const;

  private:
    const std::vector<City> m_cities;
    const DistanceTable m_distances;
};


inline double World::getDistance(const int i, const int j) const
{
  return m_distances.getDistance(i, j);
}

