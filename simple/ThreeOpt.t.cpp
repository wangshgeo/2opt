#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "SimpleSolver.h"
#include "Tour.h"


inline int optimize(const char* filename)
{
    Reader r(filename);
    const DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCities().size());
    SimpleSolver s(t.getCityCount());
    s.optimize(d, t);
    std::cout << t.length(d) << std::endl;
    return t.length(d);
}

int main(int argc, char* argv[])
{
    assert(optimize("../sets/burma14.tsp") == 3323);
    assert(optimize("../sets/ulysses16.tsp") == 6859);
    assert(optimize("../sets/berlin52.tsp") == 7542);
    // assert(optimize("../sets/kroA100.tsp") == 21282);
    // assert(optimize("../sets/ch150.tsp") == 6528);
    // assert(optimize("../sets/gr202.tsp") == 40160);

    std::cout << "Passed all tests." << std::endl;
    return 0;
}


