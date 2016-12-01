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

    void exchange();
    void optimize();
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
    void determineCityIds(const int si, const int sj, const int sk) const;
    int calculateNewCost(const ExchangeType) const;
    bool isNewSegment(const int currIndex, const int cityId) const;
};


