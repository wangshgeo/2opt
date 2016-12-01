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


void ThreeOpt::optimize()
{
    identify();
    while(m_change < 0)
    {
        const int before = m_t.length(m_d);
        printTour(m_t);
        exchange();
        printTour(m_t);
        const int after = m_t.length(m_d);
        std::cout << "delta: " << after-before << std::endl;
        identify();
        assert(m_t.valid());
        std::cout << "iter end change: " << m_change << std::endl;
    }
}


bool ThreeOpt::isNewSegment(const int currIndex, const int cityId) const
{
    std::cout << "prev, next, other: "
        << m_prev[currIndex] << ", "
        << m_next[currIndex] << ", "
        << cityId << std::endl;
    return m_next[currIndex] != cityId and m_prev[currIndex] != cityId;
}


int ThreeOpt::calculateNewCost(const ExchangeType e) const
{
    // Check for existing edge and abort if found
    //  (that would be 2opt and we don't do 2opt here).
    switch(e)
    {
        case ExchangeType::I:
            if(isNewSegment(2, m_next[0]))
            {
                return m_d.getDistance(m_curr[0], m_curr[1])
                    + m_d.getDistance(m_curr[2], m_next[0])
                    + m_d.getDistance(m_next[1], m_next[2]);
            }
            break;
        case ExchangeType::J:
            if(isNewSegment(0, m_next[1]))
            {
                return m_d.getDistance(m_curr[0], m_next[1])
                    + m_d.getDistance(m_curr[1], m_curr[2])
                    + m_d.getDistance(m_next[0], m_next[2]);
            }
            break;
        case ExchangeType::TRIPLE:
            if(isNewSegment(0, m_next[1])
                and isNewSegment(1, m_next[2])
                and isNewSegment(2, m_next[0]))
            {
                return m_d.getDistance(m_curr[0], m_next[1])
                    + m_d.getDistance(m_curr[1], m_next[2])
                    + m_d.getDistance(m_curr[2], m_next[0]);
            }
            break;
        case ExchangeType::K:
            if(isNewSegment(1, m_next[2]))
            {
                return m_d.getDistance(m_curr[0], m_curr[2])
                    + m_d.getDistance(m_curr[1], m_next[2])
                    + m_d.getDistance(m_next[0], m_next[1]);
            }
            break;
        default:
            break;
    }
    std::cout << "invalid swap set "
        << m_curr[0] << ", "
        << m_curr[1] << ", "
        << m_curr[2] << "; "
        << static_cast<char>(e) << std::endl;
    return std::numeric_limits<int>::max();
}


void ThreeOpt::identify() const
{
    for(int si = 4; si < m_t.getCityCount(); ++si)
    {
        int sj = (si == m_t.getCityCount() - 1) ? 3: 2;
        for(; sj < si - 1; ++sj)
        {
            int sk = (si == m_t.getCityCount() - 1) ? 1: 0;
            for(; sk < sj - 1; ++sk)
            {
                determineCityIds(si, sj, sk);
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


void ThreeOpt::exchange()
{
    switch(m_e)
    {
        case ExchangeType::I:
            m_t.exchange(m_s[0], m_s[1]);
            m_t.exchange(m_s[1], m_s[2]);
        break;
        case ExchangeType::J:
            m_t.exchange(m_s[1], m_s[2]);
            m_t.exchange(m_s[2], m_s[0]);
        break;
        case ExchangeType::TRIPLE:
            m_t.exchange(m_s[0], m_s[1]);
            m_t.exchange(m_s[1], m_s[2]);
            m_t.exchange(m_s[0], m_s[2]);
        break;
        case ExchangeType::K:
            m_t.exchange(m_s[2], m_s[0]);
            m_t.exchange(m_s[1], m_s[2]);
        break;
        default:
        break;
    }
}


void ThreeOpt::determineCityIds(const int si, const int sj, const int sk) const
{
    m_prev[0] = m_t.getPrevCityId(si);
    m_prev[1] = m_t.getPrevCityId(sj);
    m_prev[2] = m_t.getPrevCityId(sk);
    m_curr[0] = m_t.getCityId(si);
    m_curr[1] = m_t.getCityId(sj);
    m_curr[2] = m_t.getCityId(sk);
    m_next[0] = m_t.getNextCityId(si);
    m_next[1] = m_t.getNextCityId(sj);
    m_next[2] = m_t.getNextCityId(sk);
}


