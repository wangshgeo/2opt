#pragma once

#include "World.h"

class SimpleSolver
{
  public:
    struct Solution
    {
      double change;
      int i, j;
    };
    void optimize(World& w);
    Solution identify(const World& w);
    void improve(World& w);
  private:
    Solution currentBest;
};

