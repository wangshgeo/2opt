#pragma once

#include <cmath>

#include <vector>

#include "City.h"

class World
{
  public:
    void computeDistances()
    {
      for(size_t i = 1; i < cities.size(); ++i)
      {
        for(size_t j = 0; j < i; ++j)
        {
          distances[serialize(i, j)] = distance(i, j);
        }
      }
    }
    inline double distance(const int i, const int j) const
    {
      const double dx = cities[i].x - cities[j].x;
      const double dy = cities[i].y - cities[j].y;
      return std::sqrt(dx * dx + dy * dy);
    }
    // Assumes i > j and i > 0.
    // This converts the pairing of i and j to a unique index
    //  that has a contiguous range of values.
    inline int serialize(const int i, const int j) const
    {
      return serializedStart(i) + j;
    }
  private:
    std::vector<City> cities;
    std::vector<double> distances;
    std::vector<int> starts;

    inline bool odd(const int i) const { return i & static_cast<const int>(1); }
    // Returns the starting serialized index, given i.
    inline int serializedStart(const int i) const
    {
      return ((i * i) >> 1) - (i >> 1);
    }
};
