#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

#include "CImg.h"
#include "image.h"

#include "../Instance.h"

using namespace std;

int main(int argc,char**argv)
{
	if(argc < 2)
	{
		cout << "Please input a file name." << endl;
		return EXIT_SUCCESS;
	}
	string output_name("tour");
	if(argc > 2) output_name = argv[2];
	string file_name(argv[1]);

	Instance instance(file_name);

	fprintf(stdout, "Number of bodies: %d\n", instance.cities());

	double* x = instance.x();
	double* y = instance.y();

	write_static_image(output_name, x, y, instance.cities());
	
	return EXIT_SUCCESS;
}