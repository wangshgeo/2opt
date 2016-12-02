#pragma once

#include <cassert>
#include <cstdlib>

#include <array>
#include <iostream>
#include <limits>

#include "DistanceTable.h"
#include "Tour.h"

class ThreeOpt
{
public:
    ThreeOpt(const DistanceTable& d, Tour& t);

    inline int optimize();
private:
    const DistanceTable& m_d;
    Tour& m_t;
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


int ThreeOpt::optimize()
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


bool ThreeOpt::isNewSegment(const int currIndex, const int cityId) const
{
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
    return std::numeric_limits<int>::max();
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


void ThreeOpt::determineCityIds(const int s0, const int s1, const int s2) const
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


