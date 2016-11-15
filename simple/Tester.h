#pragma once

#include "World.h"

class Tester : private World
{
  public:
    Tester() = delete;
    Tester(const World& w) : m_world(w) {}
    void testSerialize(const int i, const int j, const int expected) const
    {
      assert(m_world.serialize(i, j) == expected);
    }
    void testDistance();
  private:
    const World& m_world;
};


