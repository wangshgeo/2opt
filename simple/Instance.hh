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

#include "types.hh"

class Instance
{
public:
	dtype* getX() const { return x; }
	dtype* getY() const { return y; }
	dtype getDistance(int n1,int n2) const { return table[n1][n2]; }
	// dtype getAngle(int n1,int n2) const { return angles[n1][n2]; }
	int getCityCount() const { return nb_cities; }
	//Constructor
	Instance(char*coordinateFile,bool precompute_=true) : precompute(precompute_) { instanceFromFile(coordinateFile); }
	//Destructor
	~Instance() { deallocateTable();delete[] x;delete[] y; }
private:
	//variables.
	int nb_cities;
	dtype *x,*y;
	dtype **table;
	dtype **angles;
	bool precompute;
	//for branch and bound or other methods utilizing geometry
	void constructAngles(dtype*x,dtype*y);
	//distance functions.
	void constructDistanceTable(dtype*x,dtype*y,int df);
	dtype coordinateToLongitudeLatitude(dtype coordinate) const;
	dtype distanceGEO(dtype*x,dtype*y,int n1,int n2) const;
	dtype distanceEUC_2D(dtype*x,dtype*y,int n1,int n2) const;
	//for reading input files.
	int readCityCount(char*coordinateFile) const;
	void readCityCoordinates(char*filepath,dtype*x,dtype*y) const;
	int readEdgeWeightType(char*filepath) const;
	vector<string> spaceTokens(string line) const;
	//for constructor
	void instanceFromFile(char*coordinateFile);
	//for destructor
	void deallocateTable();
};

#endif
