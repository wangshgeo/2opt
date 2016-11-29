#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "TwoOpt.h"
#include "Tour.h"


inline int optimize(const char* filename)
{
    Reader r(filename);
    const DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCities().size());
    TwoOpt s(t.getCityCount());
    s.optimizeParallel(d, t);
    return t.length(d);
}


void printResult(const int result, const int optimal)
{
    std::cout << "Result: " << result <<
        " (Optimal: " << optimal << ")" << std::endl;
}


int main(int argc, char* argv[])
{
    printResult(optimize("../sets/burma14.tsp"), 3323);
    printResult(optimize("../sets/ulysses16.tsp"), 6859);
    printResult(optimize("../sets/berlin52.tsp"), 7542);
    // assert(optimize("../sets/kroA100.tsp") == 21282);
    // assert(optimize("../sets/ch150.tsp") == 6528);
    // assert(optimize("../sets/gr202.tsp") == 40160);

    return 0;
}


