#include "SimpleSolver.h"

void SimpleSolver::identify(const World& world)
{
  for(int i = 1; i < world.getCityCount(); ++i)
  {
    for(int j = 0; j < i; ++j)
    {
      const double currentCost = world.getDistance(i, i + 1) + world.getDistance(j, j + 1);
      const double newCost = world.getDistance(i, j + 1) + world.getDistance(j, i + 1);
      const double change = newCost - currentCost;
      if(change < currentBest.change) currentBest = {change, i, j};
    }
  }
}
void SimpleSolver::improve(World& world) const
{

}
void SimpleSolver::optimize(World& world)
{
  identify(world);
  while(currentBest.change < 0)
  {
    improve(world);
    identify(world);
  }
}

