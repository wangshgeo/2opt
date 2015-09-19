#ifndef __MAIN_H__
#define __MAIN_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h> 
#include <math.h>
using namespace std;

#include "types.h"

void deallocateArrays();
void displayCoordinates();
coordType tourCost(int *tour, int nt);
coordType nodeDistance(int n1,int n2);
void initMask();




#endif