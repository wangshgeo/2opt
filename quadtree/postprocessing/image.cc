#include "image.h"

void write_static_image(const double*x, const double*y, const int n )
{
	CImg<float> image(IMAGE_SIZE,IMAGE_SIZE,1,3,0);
	const float color[] = {255,255,255};
	const float color_red[] = {255,0,0};

	double xmin=x[0];
	double xmax=x[0];
	double ymin=y[0];
	double ymax=y[0];
	for(int i=1;i<n;++i)
	{
		xmin = ( x[i] < xmin ) ? x[i] : xmin;
		xmax = ( x[i] > xmax ) ? x[i] : xmax;
		ymin = ( y[i] < ymin ) ? y[i] : ymin;
		ymax = ( y[i] > ymax ) ? y[i] : ymax;
	}
	double xrange = xmax-xmin;
	double yrange = ymax-ymin;

	double maxrange = (xrange > yrange) ? xrange : yrange;
	double view_dimension = 0.05*maxrange;

	xmin -= view_dimension;
	ymin -= view_dimension;
	xmax += view_dimension;
	ymax += view_dimension;
	xrange += 2*view_dimension;
	yrange += 2*view_dimension;

	image.draw_line(0,IMAGE_SIZE/2,IMAGE_SIZE,IMAGE_SIZE/2,color_red);
	image.draw_line(IMAGE_SIZE/2,0,IMAGE_SIZE/2,IMAGE_SIZE,color_red);

	for(int i=0;i<n;++i)
	{
		int px = (x[i] - xmin) / xrange * IMAGE_SIZE;
		int py = IMAGE_SIZE - (y[i] - ymin) / yrange * IMAGE_SIZE;
		image.draw_circle(px,py,BODY_RADIUS,color);
		char num[100];
		sprintf ( num, "%d", i+1 );
		image.draw_text(px,py,num,color);
		int px_next = (x[(i+1) % n] - xmin) / xrange * IMAGE_SIZE;
		int py_next = IMAGE_SIZE - (y[(i+1) % n] - ymin) / yrange * IMAGE_SIZE;
		image.draw_line(px,py,px_next,py_next,color);
	}

	std::string prefix("output/tour.png");
	image.save(prefix.c_str());
}
