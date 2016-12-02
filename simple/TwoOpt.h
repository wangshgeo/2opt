#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "DistanceTable.h"
#include "Tour.h"


class TwoOpt
{
public:
    TwoOpt(const int restarts, const DistanceTable&, Tour&);

    int optimize();
    void optimizeRestarts();
    void optimizeParallel();
private:
    const int m_restarts = 10;
    const DistanceTable& d;
    Tour& t;

    struct Solution
    {
        Solution() : change(0), si(0), sj(0) {}
      int change;
      int si, sj;
    };
    Solution identify() const;
};


