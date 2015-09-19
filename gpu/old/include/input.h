
#ifndef __INPUT_H__
#define __INPUT_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h> 
using namespace std;

void readTSP(string path);
void readOptTour(string path);
vector<string> spaceTokens(string line);
int readTSPHeader(ifstream *file);
void readTSPCoordinates(ifstream *file);


void allocateArrays(int size);
void fillDistanceTable();


#endif
