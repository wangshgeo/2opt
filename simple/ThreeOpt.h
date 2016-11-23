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
      int change;
      int s[3];
    };
    void optimize(const DistanceTable& d, Tour& t);
private:
    const int m_restarts = 1;

    Solution identify(const DistanceTable& d, const Tour& t) const;
};


