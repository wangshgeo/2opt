#pragma once

#include <cstdlib>

#include "DistanceTable.h"
#include "Tour.h"


class SimpleSolver
{
public:
    struct Solution
    {
      double change;
      int si, sj;
    };
    void optimize(const DistanceTable& d, Tour& t);
private:
    const int m_restarts = 1000;

    Solution identify(const DistanceTable& d, const Tour& t) const;
    void perturb(Tour& t);
};


