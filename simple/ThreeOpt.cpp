#include "ThreeOpt.h"


ThreeOpt::Solution ThreeOpt::identify(const DistanceTable& d, const Tour& t) const
{
    Solution bestChange;
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
                const int* cheapest = std::min_element(
                    newCosts.begin(), newCosts.end());
                const int currentCost = d.getDistance(i, inext)
                    + d.getDistance(j, jnext)
                    + d.getDistance(k, knext);
                const int change = *cheapest - currentCost;
                if(change < bestChange.change)
                {
                    bestChange.change = change;
                    bestChange.s[0] = sk;
                    bestChange.s[1] = sj;
                    bestChange.s[2] = si;
                    bestChange.e = [&]()
                    {
                        switch(cheapest - newCosts.begin())
                        {
                            case 0: return Solution::ExchangeType::I;
                            case 1: return Solution::ExchangeType::J;
                            case 2: return Solution::ExchangeType::TRIPLE;
                            case 3: return Solution::ExchangeType::K;
                            default: return Solution::ExchangeType::NONE;
                        }
                    }();
                }
            }
        }
    }
    std::cout << "Best change: " << bestChange.change
        << "; i, j, k: " << bestChange.s[0]
        << ", " << bestChange.s[1]
        << ", " << bestChange.s[2]
        << "; type: " << static_cast<char>(bestChange.e)
        << std::endl;
    return bestChange;
}


void ThreeOpt::exchange(const Solution& s, Tour& t)
{
    switch(s.e)
    {
        case Solution::ExchangeType::I:
            t.exchange(s.s[0], s.s[1]);
            t.exchange(s.s[1], s.s[2]);
        break;
        case Solution::ExchangeType::J:
            t.exchange(s.s[1], s.s[2]);
            t.exchange(s.s[2], s.s[0]);
        break;
        case Solution::ExchangeType::TRIPLE:
            t.exchange(s.s[0], s.s[1]);
            t.exchange(s.s[1], s.s[2]);
            t.exchange(s.s[0], s.s[2]);
        break;
        case Solution::ExchangeType::K:
            t.exchange(s.s[2], s.s[0]);
            t.exchange(s.s[1], s.s[2]);
        break;
        default:
        break;
    }
}

void printTour(const Tour& t)
{
    for(auto x : t.getTour())
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}
void ThreeOpt::optimize(const DistanceTable& d, Tour& t)
{
    Tour best = t;
    for(int i = 0; i < m_restarts; ++i)
    {
        Solution s = identify(d, t);
        while(s.change < 0)
        {
            const int before = t.length(d);
            printTour(t);
            exchange(s, t);
            printTour(t);
            const int after = t.length(d);
            std::cout << "delta: " << after-before << std::endl;
            s = identify(d, t);
            assert(t.valid());
        }
        if(t.length(d) < best.length(d))
        {
            best = t;
        }
        t.shuffle();
    }
    t = best;
}


