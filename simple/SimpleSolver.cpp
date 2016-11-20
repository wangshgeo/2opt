#include "SimpleSolver.h"


SimpleSolver::SimpleSolver(const Tour& t) : m_bestTour(t.getTour())
{

}


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
    m_bestTour = t;
    for(int i = 0; i < m_restarts; ++i)
    {
        Solution trial = identify(d, t);
        while(trial.change < 0)
        {
            t.exchange(trial.si, trial.sj);
            trial = identify(d, t);
        }
        if(t.length(d) < m_bestTour.length(d))
        {
            m_bestTour = t;
        }
        else
        {
            t = m_bestTour;
        }
        perturb(t);
    }
    t = m_bestTour;
}


void SimpleSolver::perturb(Tour& t)
{
    const int i = std::rand() % t.getCityCount();
    const int j = std::rand() % t.getCityCount();
    t.exchange(i, j);
}


