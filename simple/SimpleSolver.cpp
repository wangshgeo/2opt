#include "SimpleSolver.h"



SimpleSolver::Solution SimpleSolver::identify(const DistanceTable& d, const Tour& t) const
{
    assert(t.valid());
    Solution bestChange = {0, 0, 0};
    for(int si = 2; si < t.getCityCount(); ++si)
    {
        int sj = (si == t.getCityCount() - 1) ? 1 : 0;
        for(; sj < si - 1; ++sj)
        {
            const int i = t.getCityId(si);
            const int j = t.getCityId(sj);
            const int inext = t.getNextCityId(si);
            const int jnext = t.getNextCityId(sj);
            const int currentCost = d.getDistance(i, inext) + d.getDistance(j, jnext);
            const int newCost = d.getDistance(i, j) + d.getDistance(inext, jnext);
            const int change = newCost - currentCost;
            if(change == 0)
            {
                std::cout << i << " " << j << std::endl;
                std::cout << inext << " " << jnext << std::endl;
                std::cout << si << " " << sj << std::endl;
                std::cout << newCost << " " << currentCost << std::endl;
                std::cout << "---" << std::endl;
            }
            if(change < bestChange.change)
            {
                bestChange = {change, si, sj};
            }
        }
    }
    return bestChange;
}


void SimpleSolver::optimize(const DistanceTable& d, Tour& t)
{
    Tour best = t;
    assert(best.valid());
    for(int i = 0; i < m_restarts; ++i)
    {
        t.shuffle();
        Solution s = identify(d, t);
        while(s.change < 0)
        {
            t.exchange(s.si, s.sj);
            s = identify(d, t);
        }
        if(t.length(d) < best.length(d))
        {
            best = t;
        }
        else
        {
            t = best;
        }
    }
    t = best;
}


void SimpleSolver::perturb(Tour& t)
{
    const int i = std::rand() % t.getCityCount();
    const int j = std::rand() % t.getCityCount();
    t.exchange(i, j);
}


