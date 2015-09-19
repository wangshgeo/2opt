#ifndef __DRAWING_H__
#define __DRAWING_H__

#include "types.h"
#include "CImg.h"
using namespace cimg_library;

void fillConstants();
void drawCities();
void drawConvexHull();
coordType getMaxX();
coordType getMinX();
coordType getMaxY();
coordType getMinY();
int xtoc(coordType x);
int ytor(coordType y);
void drawCity(int n,CImg<unsigned char>* visu,const unsigned char* color);
void displayCurrentHull(CImg<unsigned char>* visu,const unsigned char* color);


#endif