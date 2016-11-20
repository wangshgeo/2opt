#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "SimpleSolver.h"
#include "Tour.h"

int main(int argc, char* argv[])
{
    Reader r("worlds/burma14.tsp");
    DistanceTable d(r.getCities(), DistanceTable::DistanceFunction::GEO);
    Tour t(r.getCities().size());
    SimpleSolver s;
    s.optimize(d, t);
    std::cout << t.length(d) << std::endl;

    std::cout << "Passed all tests." << std::endl;
    return 0;
}


