#include "SimpleSolver.h"


void SimpleSolver::identify(const DistanceTable& d, const Tour& t)
{
    currentBest = {0, 0, 0};
    for(int si = 0; si < t.getCityCount() - 2; ++si)
    {
        for(int sj = si + 2; sj < t.getCityCount(); ++sj)
        {
            const int i = t.getCityId(si);
            const int j = t.getCityId(sj);
            const int inext = t.getCityId(si + 1);
            const int jnext = t.getCityId(sj + 1);
            const double currentCost = d.getDistance(i, inext) + d.getDistance(j, jnext);
            const double newCost = d.getDistance(i, jnext) + d.getDistance(j, inext);
            const double change = newCost - currentCost;
            if(change < currentBest.change) currentBest = {change, si, sj};
        }
    }
}


#include <iostream>
void SimpleSolver::optimize(const DistanceTable& d, Tour& t)
{
    identify(d, t);
    while(currentBest.change < 0)
    {
        if(currentBest.change < 0)
        {
            t.exchange(currentBest.si, currentBest.sj);
        }
        identify(d, t);
        std::cout << currentBest.change << ", "
            << currentBest.si << ", " << currentBest.sj << std::endl;
    }
}

