#include <cassert>
#include <iostream>
#include <vector>

#include "Tour.h"


int main(int argc, char* argv[])
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

    std::cout << "Tests passed." << std::endl;
    return 0;
}
