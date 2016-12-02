#include "TwoOpt.h"


TwoOpt::TwoOpt(const int restarts, const DistanceTable& d, Tour& t)
    : m_restarts(restarts), d(d), t(t) {}


TwoOpt::Solution TwoOpt::identify() const
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


int TwoOpt::optimize()
{
    int iterations = 0;
    Solution s = identify();
    while(s.change < 0)
    {
        t.exchange(s.si, s.sj);
        s = identify();
        ++iterations;
    }
    return iterations;
}


void TwoOpt::optimizeRestarts()
{
    Tour best = t;
    for(int i = 0; i < m_restarts; ++i)
    {
        Solution s = identify();
        while(s.change < 0)
        {
            t.exchange(s.si, s.sj);
            s = identify();
        }
        if(t.length(d) < best.length(d))
        {
            best = t;
        }
        t.shuffle();
    }
    t = best;
}


void TwoOpt::optimizeParallel()
{
    Tour currentTour = t;
    #pragma omp parallel for firstprivate(currentTour)
    for(int i = 0; i < m_restarts; ++i)
    {
        #pragma omp critical
        currentTour.shuffle();
        Solution s = identify();
        while(s.change < 0)
        {
            currentTour.exchange(s.si, s.sj);
            s = identify();
        }
        const int currentLength = currentTour.length(d);
        #pragma omp critical
        if(currentLength < t.length(d))
        {
            t = currentTour;
        }
    }
}



