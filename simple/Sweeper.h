#pragma once

#include <cassert>
#include <cstdlib>

#include <iostream>
#include <limits>

#include "DistanceTable.h"
#include "ThreeOpt.h"
#include "Tour.h"
#include "TwoOpt.h"

class Sweeper
{
public:
    Sweeper(const DistanceTable& d, Tour& t);

    inline void optimize();
private:
    TwoOpt      two;
    ThreeOpt    three;

    inline int sweep();
};


inline void Sweeper::optimize()
{
    int iterations = sweep();
    int sweeps = 1;
    while(iterations > 0)
    {
        iterations = sweep();
        ++sweeps;
    }
    std::cout << "sweeps: " << sweeps << std::endl;
}


inline int Sweeper::sweep()
{
    int iterations = two.optimize();
    iterations += three.optimize();
    return iterations;
}


