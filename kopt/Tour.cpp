#include "Tour.h"


Tour::Tour(const std::size_t cities, const DistanceTable& d)
    : m_d(d)
{
    initialize(cities);
    connect();
    computeDistances();
}


bool Tour::valid() const
{
    std::vector<int> visited(m_s.size(), 0);
    for(const auto s : m_s)
    {
        if(++visited[s.c[0]] > 2 or ++visited[s.c[1]] > 2)
        {
            return false;
        }
    }
    return true;
}


void Tour::print() const
{
    int prevCity = m_s[0].c[0];
    const Segment* const firstSegment = &m_s[0];
    const Segment* currentSegment = firstSegment;
    do
    {
        std::cout << prevCity << " ";
        int nextCity = currentSegment->nextCity(prevCity);
        currentSegment = currentSegment->nextSegment(nextCity);
        prevCity = nextCity;
    } while (currentSegment != firstSegment);
    std::cout << "\n";
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


void Tour::computeDistances()
{
    for(auto& s : m_s)
    {
        s.length = m_d.get(s.c[0], s.c[1]);
    }
}


