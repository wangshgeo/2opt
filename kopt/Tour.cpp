#include "Tour.h"


Tour::Tour(const std::size_t cities, const DistanceTable& d)
{
    initialize(cities);
    connect();
    computeDistances(d);
}


bool Tour::valid() const
{
    std::vector<int> visited(m_s.size(), 0);
    for(const auto s : m_s)
    {
        if(++visited[s[0]] > 2 or ++visited[s[1]] > 2)
        {
            return false;
        }
    }
    return true;
}


int Tour::length() const
{
    int sum = 0;
    for (const auto s : m_s)
    {
        sum += s.length;
    }
    return sum;
}


void Tour::initialize(const std::size_t cities)
{
    m_s.resize(cities);
    for(std::size_t i = 0; i < m_s.size(); ++i)
    {
        m_s[i].c[0] = i;
        m_s[i].c[1] = (i + 1) % m_s.size();
    }
}


void Tour::connect()
{
    Segment* prev = &m_s.back();
    for(std::size_t i = 0; i < m_s.size(); ++i)
    {
        Segment* next = &m_s[(i + 1) % m_s.size()];
        m_s[i].s[0] = prev;
        m_s[i].s[1] = next;
        prev = &m_s[i];
    }
}


void Tour::compute(const DistanceTable& d)
{
    for(auto& s : m_s)
    {
        s.length = d.get(s[0], s[1]);
    }
}


