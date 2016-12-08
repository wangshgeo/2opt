#include <cassert>
#include <iostream>

#include "IndexHash.h"


int main(int argc, char* argv[])
{
    constexpr int cityCount = 10;
    IndexHash h(cityCount);
    int hashed = -1;
    for(int i = 1; i < cityCount; ++i)
    {
        for(int j = 0; j < i; ++j)
        {
            assert(h.hash(i, j) == ++hashed);
        }
    }

    std::cout << "Passed all tests." << std::endl;
    return 0;
}
