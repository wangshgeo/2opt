#include "Sweeper.h"


Sweeper::Sweeper(const DistanceTable& d, Tour& t)
    : two(1, d, t), three(d, t) {}



