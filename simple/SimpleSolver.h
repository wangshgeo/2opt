#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "DistanceTable.h"
#include "Tour.h"


class SimpleSolver
{
public:
    struct Solution
    {
      int change;
      int si, sj;
    };
    void optimize(const DistanceTable& d, Tour& t);
private:
    const int m_restarts = 10;

    Solution identify(const DistanceTable& d, const Tour& t) const;
};


