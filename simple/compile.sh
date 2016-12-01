#!/bin/sh

g++ -std=c++11 TwoOpt.t.cpp TwoOpt.cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp -o 2opt
g++ -std=c++11 -fopenmp TwoOptParallel.t.cpp TwoOpt.cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp -o 2optParallel

g++ -std=c++11 -O3 -Wall ThreeOpt.t.cpp ThreeOpt.cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp -o 3opt
