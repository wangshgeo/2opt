#pragma once

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
    void identify(const DistanceTable& d, const Tour& t);
  private:
    Solution currentBest;
};

