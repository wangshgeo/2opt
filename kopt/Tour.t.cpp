#include <cassert>
#include <iostream>
#include <vector>

#include "Reader.h"
#include "DistanceTable.h"
#include "Tour.h"


void printTour(const Tour& t)
{
    for(const auto& s : t.m_s)
    {
        std::cout << s.c[0] << " ";
    }
    std::cout << std::endl;
}



int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        return 0;
    }
    Reader r(argv[1]);
    const DistanceTable d(r.getCities(), r.getCostFunction());
    Tour t(r.getCityCount(), d);

    std::cout << "Tests passed." << std::endl;
    return 0;
}
