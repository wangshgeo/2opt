// #include "include/input.h"
// #include "include/draw.h"
// #include "include/distance.h"

int main(int argc, char* argv[])
{
	CImg<unsigned char> visu=makeVisualizer();
	drawCities(x,y,nb_cities,&visu);
	if(argc > 2) //tour input, under construction
	{
		//cout << "argc: " << argc << "\n";
		readTour(argv[2],tour);
		//drawTour(x,y,tour,nb_cities,&visu);
		drawTourSteps(x,y,tour,nb_cities,&visu);
	}
	display(&visu);
}
