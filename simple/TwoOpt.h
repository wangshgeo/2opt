#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "DistanceTable.h"
#include "Tour.h"


class TwoOpt
{
public:
    TwoOpt(const int restarts) : m_restarts(restarts) {}
    struct Solution
    {
        Solution() : change(0), si(0), sj(0) {}
      int change;
      int si, sj;
    };
    void optimize(const DistanceTable& d, Tour& t);
    void optimizeParallel(const DistanceTable& d, Tour& t);
private:
    const int m_restarts = 10;

    Solution identify(const DistanceTable& d, const Tour& t) const;
};


