#include "SimpleSolver.h"



SimpleSolver::Solution SimpleSolver::identify(const DistanceTable& d, const Tour& t) const
{
    Solution bestChange = {0, 0, 0};
    for(int si = 0; si < t.getCityCount() - 2; ++si)
    {
        for(int sj = si + 2; sj < t.getCityCount(); ++sj)
        {
            const int i = t.getCityId(si);
            const int j = t.getCityId(sj);
            const int inext = t.getNextCityId(si);
            const int jnext = t.getNextCityId(sj);
            const double currentCost = d.getDistance(i, inext) + d.getDistance(j, jnext);
            const double newCost = d.getDistance(i, j) + d.getDistance(inext, jnext);
            const double change = newCost - currentCost;
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


