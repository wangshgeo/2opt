#include "SimpleSolver.h"

void SimpleSolver::identify(const World& world)
{
  currentBest = {0, 0, 0};
  for(int si = 1; si < world.getCityCount() - 1; ++si)
  {
    for(int sj = 0; sj < si - 1; ++sj)
    {
      const int i = world.getCityId(si);
      const int j = world.getCityId(sj);
      const int inext = world.getCityId(si + 1);
      const int jnext = world.getCityId(sj + 1);
      const double currentCost = world.getDistance(i, inext) + world.getDistance(j, jnext);
      const double newCost = world.getDistance(i, jnext) + world.getDistance(j, inext);
      const double change = newCost - currentCost;
      if(change < currentBest.change) currentBest = {change, si, sj};
    }
  }
  const int si = world.getCityCount() - 1;
  for(int sj = 0; sj < si - 1; ++sj)
  {
    const int i = world.getCityId(si);
    const int j = world.getCityId(sj);
    const int inext = world.getCityId(0);
    const int jnext = world.getCityId(sj + 1);
    const double currentCost = world.getDistance(i, inext) + world.getDistance(j, jnext);
    const double newCost = world.getDistance(i, jnext) + world.getDistance(j, inext);
    const double change = newCost - currentCost;
    if(change < currentBest.change) currentBest = {change, si, sj};
  }
}
void SimpleSolver::improve(World& world) const
{
  world.reverse(currentBest.si, currentBest.sj);
}
#include <iostream>
void SimpleSolver::optimize(World& world)
{
  identify(world);
  while(currentBest.change < 0)
  {
    improve(world);
    identify(world);
    std::cout << currentBest.change << ", "
      << currentBest.si << ", " << currentBest.sj << std::endl;
  }
}

