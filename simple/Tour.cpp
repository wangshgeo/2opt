#include "Tour.h"


Tour::Tour(const int cityCount) : m_tour(cityCount, 0)
{
    for(int i = 0; i < cityCount; ++i)
    {
        m_tour[i] = i;
    }
}


Tour::Tour(std::vector<int> initial) : m_tour(std::move(initial)) {}


bool Tour::valid() const
{
    std::vector<bool> visited(m_tour.size(), false);
    for(auto city : m_tour)
    {
        bool validIndex = city >= 0 and city < static_cast<int>(m_tour.size());
        if(not validIndex)
        {
            return false;
        }
        else
        {
            if(visited[city])
            {
                // premature cycle.
                return false;
            }
            else
            {
                visited[city] = true;
            }
        }
    }
    return true;
}


double Tour::length(const DistanceTable& d) const
{
    double total = 0;
    for(size_t i = 0; i < m_tour.size() - 1; ++i)
    {
        total += d.getDistance(m_tour[i], m_tour[i + 1]);
    }
    total += d.getDistance(m_tour.back(), m_tour.front());
    return total;
}
