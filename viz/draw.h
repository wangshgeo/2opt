#pragma once

#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "CImg.h"

using namespace cimg_library;

class Drawer
{
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
	Drawer(char*cf)
  {
    //vv = makeVisualizer();
		coordinatesFromFile(cf);
	}
private:
	//CImg<unsigned char> vv;
	coordType xmin,ymax,yrange,xdomain;
	//calculations
	coordType getMaxX();
	coordType getMinX();
	coordType getMaxY();
	coordType getMinY();
	int xtoc(coordType x);
	int ytor(coordType y);
};

