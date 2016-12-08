#include <cassert>
#include <iostream>

#include "DistanceTable.h"
#include "Reader.h"
#include "KOpt.h"
#include "Tour.h"

inline int optimize(const char* filename)
{
    Reader r(filename);
    const DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCities().size());
    KOpt<2> s(d, t);
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

    return 0;
}


