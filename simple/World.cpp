#include "World.h"


World::World(std::vector<City> cities)
    : m_cities(std::move(cities)), m_distances(m_cities),
        m_tour(m_cities.size(), 0)
{
    for(std::size_t i = 0; i < m_tour.size(); ++i)
    {
        m_tour[i] = i;
    }
}




