#ifndef __DRAWING_H__
#define __DRAWING_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include "types.h"
#include "CImg.h"
#include "../instance.h"
using namespace cimg_library;

class Drawer
{
private:
	//CImg<unsigned char> vv;
	int nodes;
	coordType*x;
	coordType*y;
	coordType xmin,ymax,yrange,xdomain;
	//calculations
	coordType getMaxX();
	coordType getMinX();
	coordType getMaxY();
	coordType getMinY();
	int xtoc(coordType x);
	int ytor(coordType y);
	//for constructors
	vector<string> spaceTokens(string line) const;
	void coordinatesFromFile(char*coordinateFile);
	int readCityCount(char*filepath) const;
	void readCityCoordinates(char*filepath,coordType*x,coordType*y) const;
public:
	//drawing functions
	void drawCities();
	void drawTour(int*tour);
	void drawTour2(int*tour,int*path,int np);
	void drawSeq(int tour[],int seq[],int seqi);
	void drawTourSteps(int*tour);
	void display();
	//saving functions
	void saveTour(int*tour,int num);
	void saveSeq(int tour[],int seq[],int seqi,int num,int num2);
	//constructors
	Drawer(char*cf) { //vv = makeVisualizer(); 
		coordinatesFromFile(cf); 
	}
	//Drawer(Instance inst): instance(inst) { makeVisualizer(); }
	~Drawer() { delete[] x;delete[] y; }
};




//Internal stuff


#endif