#ifndef __INSTANCE_H__
#define __INSTANCE_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
using namespace std;

#include "types.h"


#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


typedef struct coord 
{
	coordType x,y;
} COORD;

class Instance
{
private:
	//variables.
	int nb_cities;
	COORD*coords;
	coordType**table;
	coordType**angles;
	//for branch and bound or other methods utilizing geometry
	void constructAngles(coordType*x,coordType*y);
	//distance functions.
	void constructDistanceTable(coordType*x,coordType*y,int df);
	coordType coordinateToLongitudeLatitude(coordType coordinate) const;
	coordType distanceGEO(coordType*x,coordType*y,int n1,int n2) const;
	coordType distanceEUC_2D(coordType*x,coordType*y,int n1,int n2) const;
	//for reading input files.
	int readCityCount(char*coordinateFile) const;
	void readCityCoordinates(char*filepath,coordType*x,coordType*y) const;
	int readEdgeWeightType(char*filepath) const;
	vector<string> spaceTokens(string line) const;
	//for constructor
	void instanceFromFile(char*coordinateFile);
	//for destructor
	void deallocateTable();
public:
	COORD*getCoords() const { return coords; }
	coordType getDistance(int n1,int n2) const { return table[n1][n2]; }
	coordType getAngle(int n1,int n2) const { return angles[n1][n2]; }
	int getCityCount() const { return nb_cities; }
	//Constructor
	Instance(char*coordinateFile) { instanceFromFile(coordinateFile); }
	//Destructor
	~Instance() { deallocateTable();delete coords; }
};

#endif
