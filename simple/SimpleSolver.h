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
    void identify(const World& w);
    void improve(World& w) const;
  private:
    Solution currentBest;
};

