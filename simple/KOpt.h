#pragma once

#include <cassert>
#include <cstdlib>

#include <array>
#include <iostream>
#include <limits>

#include "DistanceTable.h"
#include "Tour.h"
#include "../swappool/SwapPool.h"

template <std::size_t K>
class KOpt
{
public:
    KOpt(const DistanceTable& d, Tour& t);

    inline int optimize();
private:
    const DistanceTable& m_d;
    Tour& m_t;
    SwapPool<K> m_sp;
    // Current solution.
    mutable int m_change;
    mutable int m_s[3];
    mutable enum class ExchangeType : char
    {
        NONE = '\0',
        I = '0',
        J = '1',
        TRIPLE = '2',
        K = '3'
    } m_e;
    // City IDs.
    mutable int m_prev[3];
    mutable int m_curr[3];
    mutable int m_next[3];

    void identify() const;
    inline void exchange();
    inline void determineCityIds(const int si, const int sj, const int sk) const;
    inline int calculateNewCost(const ExchangeType) const;
    inline bool isNewSegment(const int currIndex, const int cityId) const;
};


template <std::size_t K>
int KOpt<K>::optimize()
{
    int iterations = 0;
    identify();
    while(m_change < 0)
    {
        exchange();
        identify();
        assert(m_t.valid());
        ++iterations;
    }
    return iterations;
}


template <std::size_t K>
bool KOpt<K>::isNewSegment(const int currIndex, const int cityId) const
{
    return m_next[currIndex] != cityId and m_prev[currIndex] != cityId;
}


template <std::size_t K>
int KOpt<K>::calculateNewCost(const ExchangeType e) const
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
    return std::numeric_limits<int>::max();
}


template <std::size_t K>
void KOpt<K>::exchange()
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


template <std::size_t K>
void KOpt<K>::determineCityIds(const int s0, const int s1, const int s2) const
{
    m_prev[0] = m_t.getPrevCityId(s0);
    m_prev[1] = m_t.getPrevCityId(s1);
    m_prev[2] = m_t.getPrevCityId(s2);
    m_curr[0] = m_t.getCityId(s0);
    m_curr[1] = m_t.getCityId(s1);
    m_curr[2] = m_t.getCityId(s2);
    m_next[0] = m_t.getNextCityId(s0);
    m_next[1] = m_t.getNextCityId(s1);
    m_next[2] = m_t.getNextCityId(s2);
}


template <std::size_t K>
KOpt<K>::KOpt(const DistanceTable& d, Tour& t) : m_d(d), m_t(t)
{
    
}


inline void printTour(const Tour& t)
{
    for(auto x : t.getTour())
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}


template <std::size_t K>
void KOpt<K>::identify() const
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


