#include "TwoOpt.h"


TwoOpt::Solution TwoOpt::identify(const DistanceTable& d, const Tour& t) const
{
    Solution bestChange;
    for(int si = 2; si < t.getCityCount(); ++si)
    {
        int sj = (si == t.getCityCount() - 1) ? 1 : 0;
        for(; sj < si - 1; ++sj)
        {
            const int i = t.getCityId(si);
            const int j = t.getCityId(sj);
            const int inext = t.getNextCityId(si);
            const int jnext = t.getNextCityId(sj);
            const int currentCost = d.getDistance(i, inext)
                + d.getDistance(j, jnext);
            const int newCost = d.getDistance(i, j)
                + d.getDistance(inext, jnext);
            const int change = newCost - currentCost;
            if(change < bestChange.change)
            {
                bestChange.change = change;
                bestChange.si = si;
                bestChange.sj = sj;
            }
        }
    }
    return bestChange;
}


void TwoOpt::optimize(const DistanceTable& d, Tour& t)
{
    Tour best = t;
    for(int i = 0; i < m_restarts; ++i)
    {
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
        t.shuffle();
    }
    t = best;
}


void TwoOpt::optimizeParallel(const DistanceTable& d, Tour& b)
{
    Tour currentTour = b;
    #pragma omp parallel for firstprivate(currentTour)
    for(int i = 0; i < m_restarts; ++i)
    {
        #pragma omp critical
        currentTour.shuffle();
        Solution s = identify(d, currentTour);
        while(s.change < 0)
        {
            currentTour.exchange(s.si, s.sj);
            s = identify(d, currentTour);
        }
        const int currentLength = currentTour.length(d);
        #pragma omp critical
        if(currentLength < b.length(d))
        {
            b = currentTour;
        }
    }
}



