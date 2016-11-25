#include "ThreeOpt.h"


ThreeOpt::Solution ThreeOpt::identify(const DistanceTable& d, const Tour& t) const
{
    Solution bestChange = {0, 0, 0, 0};
    for(int si = 2; si < t.getCityCount(); ++si)
    {
        int sj = (si == t.getCityCount() - 1) ? 1 : 0;
        for(; sj < si - 1; ++sj)
        {
            for(int sk = 0; sk < sj - 1; ++sk)
            {
                const int i = t.getCityId(si);
                const int j = t.getCityId(sj);
                const int k = t.getCityId(sk);
                const int inext = t.getNextCityId(si);
                const int jnext = t.getNextCityId(sj);
                const int knext = t.getNextCityId(sk);
                const int currentCost = d.getDistance(i, inext)
                    + d.getDistance(j, jnext)
                    + d.getDistance(k, knext);
                constexpr int PossibleArrangements = 4;
                const std::array<int, PossibleArrangements> newCosts
                {
                    d.getDistance(i, j)
                        + d.getDistance(k, inext)
                        + d.getDistance(jnext, knext),
                    d.getDistance(i, jnext)
                        + d.getDistance(j, k)
                        + d.getDistance(inext, knext),
                    d.getDistance(i, jnext)
                        + d.getDistance(j, knext)
                        + d.getDistance(k, inext),
                    d.getDistance(i, k)
                        + d.getDistance(j, knext)
                        + d.getDistance(inext, jnext)
                };
                const int cheapest = *std::min_element(
                    newCosts.begin(), newCosts.end());
                const int change = cheapest - currentCost;
                if(change < bestChange.change)
                {
                    bestChange = {change, si, sj, sk};
                }
            }
        }
    }
    return bestChange;
}


void ThreeOpt::optimize(const DistanceTable& d, Tour& t)
{
    Tour best = t;
    for(int i = 0; i < m_restarts; ++i)
    {
        Solution s = identify(d, t);
        while(s.change < 0)
        {
            t.exchange(s.s[0], s.s[1]);
            t.exchange(s.s[1], s.s[2]);
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


