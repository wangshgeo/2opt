#!/bin/sh

g++ -std=c++14 SimpleSolver.*cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp

g++ -std=c++14 ThreeOpt.*cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp
