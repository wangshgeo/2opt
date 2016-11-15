#pragma once

#include "World.h"

class Tester : private World
{
  public:
    Tester(const World& w) : w(w) {}
    void testSerialize(const int i, const int j, const int expected) const 
    {
      assert(w.serialize(i, j) == expected); 
    }
    void testDistance();
  private:
    const World& w;
};


