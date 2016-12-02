#include "ThreeOpt.h"


ThreeOpt::ThreeOpt(const DistanceTable& d, Tour& t) : m_d(d), m_t(t) {}


inline void printTour(const Tour& t)
{
    for(auto x : t.getTour())
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}


void ThreeOpt::identify() const
{
    m_change = 0;
    for(int si = 4; si < m_t.getCityCount(); ++si)
    {
        int sj = (si == m_t.getCityCount() - 1) ? 3: 2;
        for(; sj < si - 1; ++sj)
        {
            int sk = (si == m_t.getCityCount() - 1) ? 1: 0;
            for(; sk < sj - 1; ++sk)
            {
                determineCityIds(sk, sj, si);
                constexpr int PossibleArrangements = 4;
                const std::array<int, PossibleArrangements> newCosts
                {
                    calculateNewCost(ExchangeType::I),
                    calculateNewCost(ExchangeType::J),
                    calculateNewCost(ExchangeType::TRIPLE),
                    calculateNewCost(ExchangeType::K)
                };
                const int* cheapest = std::min_element(
                    newCosts.begin(), newCosts.end());
                const int currentCost = m_d.getDistance(m_curr[0], m_next[0])
                    + m_d.getDistance(m_curr[1], m_next[1])
                    + m_d.getDistance(m_curr[2], m_next[2]);
                const int change = *cheapest - currentCost;
                if(change < m_change)
                {
                    m_change = change;
                    m_s[0] = sk;
                    m_s[1] = sj;
                    m_s[2] = si;
                    m_e = [&]()
                    {
                        switch(cheapest - newCosts.begin())
                        {
                            case 0: return ExchangeType::I;
                            case 1: return ExchangeType::J;
                            case 2: return ExchangeType::TRIPLE;
                            case 3: return ExchangeType::K;
                            default: return ExchangeType::NONE;
                        }
                    }();
                }
            }
        }
    }
}


