#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "ThreeOpt.h"
#include "Tour.h"


inline int optimize(const char* filename)
{
    Reader r(filename);
    const DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCities().size());
    ThreeOpt s(d, t);
    s.optimize();
    std::cout << "Optimized length: " << t.length(d) << std::endl;
    return t.length(d);
}


int main(int argc, char* argv[])
{
    if(argc == 2)
    {
        optimize(argv[1]);
    }

    // optimize("../sets/hex6.tsp");
    // assert(optimize("../sets/burma14.tsp") == 3323);
    // assert(optimize("../sets/ulysses16.tsp") == 6859);
    // assert(optimize("../sets/berlin52.tsp") == 7542);
    // assert(optimize("../sets/kroA100.tsp") == 21282);
    // assert(optimize("../sets/ch150.tsp") == 6528);
    // assert(optimize("../sets/gr202.tsp") == 40160);

    return 0;
}


