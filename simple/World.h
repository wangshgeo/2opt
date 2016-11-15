#pragma once

#include <cmath>

#include <vector>

#include "City.h"

class World
{
  public:
    World() = default;
    World(std::vector<City> cities) : m_cities(std::move(cities))
    {
      computeStarts();
      computeDistances();
    }
    void computeStarts()
    {
      m_starts.resize(m_cities.size());
      for(size_t i = 0; i < m_cities.size(); ++i)
      {
        m_starts[i] = serializedStart(i);
      }
    }
    void computeDistances()
    {
      m_distances.resize(serializedStart(m_cities.size()));
      for(size_t i = 1; i < m_cities.size(); ++i)
      {
        for(size_t j = 0; j < i; ++j)
        {
          const int iserial = serialize(i, j);
          m_distances[iserial] = distance(i, j);
        }
      }
    }
    inline double distance(const int i, const int j) const
    {
      const double dx = m_cities[i].x - m_cities[j].x;
      const double dy = m_cities[i].y - m_cities[j].y;
      return std::sqrt(dx * dx + dy * dy);
    }
    // Assumes i > j and i > 0.
    // This converts the pairing of i and j to a unique index
    //  that has a contiguous range of values.
    inline int serialize(const int i, const int j) const
    {
      return m_starts[i] + j;
    }
    inline int getCityCount() const { return m_cities.size(); }
    inline double getDistance(const int i, const int j) const
    {
      return m_distances[serialize(i, j)];
    }
  private:
    std::vector<City> m_cities;
    std::vector<double> m_distances;
    std::vector<int> m_starts;

    // Returns the starting serialized index, given i.
    inline int serializedStart(const int i) const
    {
      return ((i * i) >> 1) - (i >> 1);
    }
};
