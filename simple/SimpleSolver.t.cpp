#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "SimpleSolver.h"
#include "Tour.h"


inline void optimize(const char* filename)
{
    Reader r(filename);
    DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCities().size());
    SimpleSolver s(t.getCityCount());
    s.optimize(d, t);
    std::cout << t.length(d) << std::endl;
}

int main(int argc, char* argv[])
{
    optimize("../sets/burma14.tsp");
    optimize("../sets/berlin52.tsp");

    std::cout << "Passed all tests." << std::endl;
    return 0;
}


