#pragma once

#include "World.h"

class SimpleSolver
{
  public:
    struct Solution
    {
      double cost;
      int i, j;
    };
    Solution iterate(const World& w);
};

