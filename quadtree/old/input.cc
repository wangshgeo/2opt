#include "input.h"


int getCityCount(char* filename)
{
	std::ifstream infile(filename);
	if(infile.peek() == std::ifstream::traits_type::eof())
	{
		fprintf(stdout, "\nCould not locate the file. Exiting.\n");
		exit(0);
	}

	std::string line;
	std::string tag("DIMENSION");
	while (infile >> line)
	{
		if(line.compare(0, tag.length(), tag) == 0)
		{
			std::string num;
			infile >> num;
			if(num.compare(0, 1, ":") == 0)
			{
				infile >> num;
			}
			return atoi(num.c_str());
		}
	}
	fprintf(stdout, "City count not found!\n");
	return 0;
}



void fill_coordinates_2D(const char* filename, dtype* x, dtype* y, const int n)
{
	std::ifstream infile(filename);
	if(infile.peek() == std::ifstream::traits_type::eof())
	{
		fprintf(stdout, "\nCould not locate the file. Exiting.\n");
		exit(0);
	}

	std::string line;
	std::string tag("NODE_COORD_SECTION");
	while (infile >> line)
	{
		if(line.compare(0, tag.length(), tag) == 0)
		{
			break;
		}
	}
	int index;
	dtype x_value, y_value;
	for(int i=0;i<n;++i)
	{
		infile >> index >> x_value >> y_value;
		x[index-1] = x_value;
		y[index-1] = y_value;
	}
}

void shuffle_cities(dtype* x, dtype* y, const int n)
{
	int *initialTour = new int[n];
	for(int i=0;i<n;++i) initialTour[i] = i;
	std::random_shuffle(initialTour,initialTour+n);
	
	dtype* x_ordered = new dtype[n];
	dtype* y_ordered = new dtype[n];
	for(int i=0;i<n;++i)
	{
		x_ordered[i]=x[initialTour[i]];
		y_ordered[i]=y[initialTour[i]];
	}
	delete[] initialTour;
	for(int i=0;i<n;++i)
	{
		x[i]=x_ordered[i];
		y[i]=y_ordered[i];
	}
	delete[] x_ordered;
	delete[] y_ordered;
}