#pragma once

#include <cmath>

#include <algorithm>
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
      initializeTour();
    }
    void initializeTour()
    {
      m_tour.resize(m_cities.size());
      for(std::size_t i = 0; i < m_tour.size(); ++i) m_tour[i] = i;
    }
    void computeStarts()
    {
      m_starts.resize(m_cities.size());
      for(std::size_t i = 0; i < m_cities.size(); ++i)
      {
        m_starts[i] = serializedStart(i);
      }
    }
    void computeDistances()
    {
      m_distances.resize(serializedStart(m_cities.size()));
      for(std::size_t i = 1; i < m_cities.size(); ++i)
      {
        for(std::size_t j = 0; j < i; ++j)
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
    // This converts the pairing of i and j to a unique index
    //  that has a contiguous range of values.
    inline int serialize(const int i, const int j) const
    {
      const int lower = (i < j) ? i : j;
      const int higher = (i > j) ? i : j;
      return m_starts[higher] + lower;
    }
    inline int getCityCount() const { return m_cities.size(); }
    inline double getDistance(const int i, const int j) const
    {
      return m_distances[serialize(i, j)];
    }
    inline void reverse(const int i, const int j)
    {
      const int lower = (i < j) ? i : j;
      const int higher = (i > j) ? i : j;
      std::reverse(m_tour.begin() + lower + 1, m_tour.begin() + higher + 1);
    }
    inline int getCityId(const int sequenceNumber) const
    {
      return m_tour[sequenceNumber];
    }
  private:
    std::vector<City> m_cities; // cityId -> position
    std::vector<double> m_distances; // hash -> distance
    std::vector<int> m_starts; // cityId -> hash start
    std::vector<int> m_tour; // order -> cityId

    // Returns the starting serialized index, given i.
    inline int serializedStart(const int i) const
    {
      return ((i * i) >> 1) - (i >> 1);
    }
};
