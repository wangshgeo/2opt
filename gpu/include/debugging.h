#ifndef __DEBUGGING_H__
#define __DEBUGGING_H__

#include "instance.h"
#include "solution.h"

void testDistanceFunctions()
{
	//Call this in main loop.
	//reads in all files and compares computed optimal distances to the known figures.
	//assumes file locations.
	string cases[] = 	{"ulysses16",	"berlin52",	"kroA100",	"ch150","gr202" };
	int opt[]=			{6859,			7542,		21282,		6528,	40160};
	string tspdir = "DATA/";
	string optdir = "OPT/";
	string tspext = ".tsp";
	string optext = ".opt.tour";
	string tspfile,optfile;
	for(int i=0;i<5;++i){
		cout << cases[i] << ": ";
		tspfile=tspdir+cases[i]+tspext;
		Instance instance(const_cast<char*>(tspfile.c_str()));
		optfile=optdir+cases[i]+optext;
		Solution solution(const_cast<char*>(optfile.c_str()),instance);
		cout << solution.getTourLength() << " (opt: " << opt[i] << ")" <<"\n";
	}
}

#endif