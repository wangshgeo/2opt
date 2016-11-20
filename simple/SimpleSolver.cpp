#include "SimpleSolver.h"

#include <iostream>

void SimpleSolver::identify(const DistanceTable& d, const Tour& t)
{
    m_currentBest = {0, 0, 0};
    for(int si = 0; si < t.getCityCount() - 2; ++si)
    {
        for(int sj = si + 2; sj < t.getCityCount(); ++sj)
        {
            const int i = t.getCityId(si);
            const int j = t.getCityId(sj);
            const int inext = t.getNextCityId(si);
            const int jnext = t.getNextCityId(sj);
            //std::cout << "i, j: " << i << ", " << j << std::endl;
            //std::cout << "inext, jnext: " << inext << ", " << jnext << std::endl;
            const double currentCost = d.getDistance(i, inext) + d.getDistance(j, jnext);
            const double newCost = d.getDistance(i, j) + d.getDistance(inext, jnext);
            const double change = newCost - currentCost;
            if(change < m_currentBest.change)
            {
                m_currentBest = {change, si, sj};
            }
        }
    }
}


void SimpleSolver::optimize(const DistanceTable& d, Tour& t)
{
    //std::cout << t.length(d) << std::endl;
    identify(d, t);
    while(m_currentBest.change < 0)
    {
        //std::cout << m_currentBest.change << ", "
        //    << m_currentBest.si << ", " << m_currentBest.sj << std::endl;
        //std::cout << t.length(d) << std::endl;
        //std::cin.ignore();
        t.exchange(m_currentBest.si, m_currentBest.sj);
        identify(d, t);
    }
}

