#include <cstdlib>
#include <fstream>
#include <string>

#include "CImg.h"
#include "types.hh"
#include "image.hh"

int main(int argc,char**argv)
{
	fprintf(stdout, "Locating data from '../output.txt'... ");
	std::string timestep_file("../output.txt");
	std::ifstream infile(timestep_file.c_str());
	if(infile.peek() == std::ifstream::traits_type::eof())
	{
		fprintf(stdout, "\nCould not locate the file. Exiting.\n");
		return EXIT_SUCCESS;
	}
	fprintf(stdout, "Done.\n");

	int n = 0;
	infile >> n;
	fprintf(stdout, "Number of bodies: %d\n",n);

	dtype* x = new dtype[n];
	dtype* y = new dtype[n];

	for(int i=0;i<n;++i) infile >> x[i] >> y[i];

	write_static_image(x,y,n);
	
	fprintf(stdout, "Done reading file.\n");

	delete[] x;
	delete[] y;

	return EXIT_SUCCESS;
}