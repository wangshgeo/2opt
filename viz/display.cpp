#include "include/types.h"
#include "include/input.h"
#include "include/draw.h"
#include "include/distance.h"

int main (int argc,char*argv[]) {
	
	char*coordinateFile=argv[1];

	int nb_cities = readCityCount(coordinateFile);
	cout << "Number of Cities: " << nb_cities << "\n";

	coordType*x=new coordType[nb_cities];
	coordType*y=new coordType[nb_cities];

	readCityCoordinates(coordinateFile,x,y,nb_cities);

	//distanceFunction df = readEdgeWeightType(coordinateFile);
	CImg<unsigned char> visu=makeVisualizer();
	drawCities(x,y,nb_cities,&visu);

	if(argc > 2)//tour input, under construction
	{
		//cout << "argc: " << argc << "\n";
		int*tour=new int[nb_cities];
		readTour(argv[2],tour);
		//drawTour(x,y,tour,nb_cities,&visu);
		drawTourSteps(x,y,tour,nb_cities,&visu);
		delete[] tour;
	}
	display(&visu);


	delete[] x;
	delete[] y;
}
