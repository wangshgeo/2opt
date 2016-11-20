#include "DistanceTable.h"


DistanceTable::DistanceTable(const std::vector<City>& cities)
    : m_distances(cities.size(), 0), m_hash(cities.size())
{
    for(std::size_t i = 1; i < cities.size(); ++i)
    {
        for(std::size_t j = 0; j < i; ++j)
        {
            const int hash = m_hash.hash(i, j);
            m_distances[hash] = distance(cities, i, j);
        }
    }
}


double DistanceTable::distance(const std::vector<City>& cities,
    const int i, const int j) const
{
  const double dx = cities[i].x - cities[j].x;
  const double dy = cities[i].y - cities[j].y;
  return std::sqrt(dx * dx + dy * dy);
}



