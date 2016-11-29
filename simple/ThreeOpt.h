#pragma once

#include <cassert>
#include <cstdlib>

#include <array>
#include <iostream>

#include "DistanceTable.h"
#include "Tour.h"


class ThreeOpt
{
public:
    ThreeOpt(const int restarts) : m_restarts(restarts) {}
    struct Solution
    {
        enum class ExchangeType : char
        {
            NONE = '\0',
            I = '0',
            J = '1',
            TRIPLE = '2',
            K = '3'
        };
        Solution() : change(0), s{0, 0, 0}, e(ExchangeType::NONE) {}
        int change;
        int s[3];
        ExchangeType e;
    };
    void exchange(const Solution& s, Tour& t);
    void optimize(const DistanceTable& d, Tour& t);
private:
    const int m_restarts = 1;

    Solution identify(const DistanceTable& d, const Tour& t) const;
};


