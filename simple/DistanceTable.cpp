#include "DistanceTable.h"


DistanceTable::DistanceTable(const std::vector<City>& cities, const DistanceFunction f)
    : m_hash(cities.size())
{
    for(std::size_t i = 1; i < cities.size(); ++i)
    {
        for(std::size_t j = 0; j < i; ++j)
        {
            switch(f)
            {
                case DistanceFunction::EUC:
                    m_distances.push_back(distance(cities, i, j));
                    break;
                case DistanceFunction::GEO:
                    m_distances.push_back(geoDistance(cities, i, j));
                    break;
                default: break;
            }
        }
    }
}


int DistanceTable::nearestInt(const double x) const
{
    return static_cast<int>(x + 0.5);
}


int DistanceTable::distance(const std::vector<City>& cities,
    const int i, const int j) const
{
  const double dx = nearestInt(cities[i].x - cities[j].x);
  const double dy = nearestInt(cities[i].y - cities[j].y);
  return nearestInt(std::sqrt(dx * dx + dy * dy));
}


double DistanceTable::toGeographic(const double d) const
{
    const int whole = static_cast<int>(d);
    const double fraction = d - whole;
    constexpr double Pi = 3.141592;
    return Pi * (whole + 5.0 * fraction / 3.0) / 180.0;
}


int DistanceTable::geoDistance(const std::vector<City>& cities,
    const int i, const int j) const
{
    constexpr double RRR = 6378.388;
    const double q1 = std::cos(
        toGeographic(cities[i].y) - toGeographic(cities[j].y));
    const double q2 = std::cos(
        toGeographic(cities[i].x) - toGeographic(cities[j].x));
    const double q3 = std::cos(
        toGeographic(cities[i].x) + toGeographic(cities[j].x));
    const double term = (1.0 + q1) * q2 - (1.0 - q1) * q3;
    return static_cast<int>(RRR * std::acos(0.5 * term) + 1.0);
}



