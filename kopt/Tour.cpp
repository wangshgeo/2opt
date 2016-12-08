#include "Tour.h"


Tour::Tour(const int cityCount)
{
    m_s.resize(cityCount);
    const int prev = cityCount - 1;
    for(int i = 0; i < cityCount; ++i)
    {
        m_s[i] = i;
    }
}


Tour::Tour(std::vector<int> initial)
{
    m_s.resize(initial.size());
    int prev = -1;
    int i = 0;
    for (auto c : initial)
    {
        m_s[i] = Segment(prev, c, )
        prev = c;
        ++i;
}


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


int Tour::length(const DistanceTable& d) const
{
    int total = 0;
    for(size_t i = 0; i < m_tour.size() - 1; ++i)
    {
        total += d.getDistance(m_tour[i], m_tour[i + 1]);
    }
    total += d.getDistance(m_tour.back(), m_tour.front());
    return total;
}



