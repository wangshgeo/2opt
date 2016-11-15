#include "SimpleSolver.h"

void SimpleSolver::identify(const World& world) const
{
  for(int i = 1; i < world.getCityCount(); ++i)
  {
    for(int j = 0; j < i; ++j)
    {
      const double currentCost = getDistance(i, i + 1) + getDistance(j, j + 1);
      const double newCost = getDistance(i, j + 1) + getDistance(j, i + 1);
      const double change = newCost - currentCost;
      if(change < currentBest.change) currentBest = {change, i, j};
    }
  }
}
void improve(World& world) const
{
  
}
void SimpleSolver::optimize(World& world) const
{
  identify(w);
  while(currentBest.change < 0)
  {
    improve(w);
    identify(w);
  }
}

