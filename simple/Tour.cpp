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
        bool validIndex = city >= 0 and city < m_tour.size();
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


