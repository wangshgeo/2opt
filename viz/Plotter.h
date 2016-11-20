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

class Plotter
{
public:
	void drawCities();
	void drawTour(int*tour);
	void drawTour2(int*tour,int*path,int np);
	void drawSeq(int tour[],int seq[],int seqi);
	void drawTourSteps(int*tour);
	void display();
	//saving functions
	void saveTour(int*tour,int num);
	void saveSeq(int tour[],int seq[],int seqi,int num,int num2);
private:
    static constexpr int CityRadius = 2;
    static constexpr int Rows = 400;
    static constexpr int Columns = 400;
    static constexpr int Padding = 20;
    static constexpr int ColumnDomain = Columns - 2 * Padding;
    static constexpr int RowRange = Rows - 2 * Padding;
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

