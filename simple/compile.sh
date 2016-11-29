#!/bin/sh

g++ -std=c++11 TwoOpt.*cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp -o 2opt

g++ -std=c++11 ThreeOpt.*cpp IndexHash.cpp DistanceTable.cpp Reader.cpp Tour.cpp -o 3opt
