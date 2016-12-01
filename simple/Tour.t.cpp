#include <cassert>
#include <iostream>
#include <vector>

#include "DistanceTable.h"
#include "Tour.h"

void printTour(const Tour& t)
{
    for(auto x : t.getTour())
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}



int main(int argc, char* argv[])
{
    {
        std::vector<int> initialTour(10);
        int i = -1;
        for(auto& city : initialTour)
        {
            city = ++i;
        }
        Tour t(initialTour);
        t.exchange(4, 9);
        t.exchange(0, 5);
        t.exchange(3, 9);
        t.exchange(0, 9);
        t.exchange(6, 6);
        t.exchange(6, 1);
        assert(t.valid());
    }
    {
        std::vector<int> initialTour(6);
        int i = -1;
        for(auto& city : initialTour)
        {
            city = ++i;
        }
        Tour t(initialTour);
        printTour(t);
        t.exchange(0, 2);
        printTour(t);
        t.exchange(2, 4);
        printTour(t);
        t.exchange(0, 4);
        printTour(t);
    }
    std::cout << "Tests passed." << std::endl;
    return 0;
}
