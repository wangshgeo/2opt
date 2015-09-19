/**
GPU 2-OPT Taveling Salesman Problem Solver
CSE6230 Final Project
Fall 2014
Robert Lee
**/

//standard libraries
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <limits>
using namespace std;

//other custom includes
#include "../include/types.h"
#include "../include/debugging.h"

//general classes
#include "../include/instance.h"
#include "../include/solution.h"

//algorithms
//#include "include/opt2.h"

int main (int argc,char*argv[]) {
	//Arguments:
	//1: Filename of dataset
	//2: Cutoff time in seconds (int).
	//3: A random seed (int).
	
	//Get arguments.
	char*coordinateFile=argv[1];
	cout << argv[1] << "\n";
	//long cutoffTime=atoi(argv[2]);
	//int randomSeed=atoi(argv[3]);

	//Declare instance (problem definition).
	Instance instance(coordinateFile);
	/*
	cout << "Starting 2-Opt." << "\n";
	Solution sol = solve2Opt(instance,cutoffTime,randomSeed);
	cout << "Finished. Tour length: " << sol.getTourLength() << " RunTime: " << sol.getRuntime() << " milliseconds. \n";
   */
	return 0;
}

