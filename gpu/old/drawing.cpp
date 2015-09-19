#include "include/main.h"
#include "include/types.h"
#include "include/drawing.h"

#include "include/geometric.h"

extern coordType *x,*y; 
extern int nodes;
extern int *optTour,*mask;
extern coordType optScore;
extern int **distanceTable;

#define CITYRADIUS 2
#define ROWS 500
#define COLUMNS 500
#define PADDING 10
#define COLUMNDOMAIN ((COLUMNS)-2*(PADDING))
#define ROWRANGE ((ROWS)-2*(PADDING))

coordType xmin,xdomain,ymax,yrange;

void fillConstants(){
	xmin = getMinX();
	xdomain=getMaxX()-xmin;
	ymax = getMaxY();
	yrange=ymax-getMinY();
}

int xtoc(coordType x)
{
	return (int)(COLUMNDOMAIN*((x-xmin)/xdomain))+PADDING;
}
int ytor(coordType y)
{
	return (int)(ROWRANGE*((ymax-y)/yrange))+PADDING;
}
void drawCities()
{
	fillConstants();


	const unsigned char black[] = {0,0,0};
	const unsigned char red[] = { 255,0,0 };
	//const unsigned char green[] = { 0,255,0 };
	//const unsigned char blue[] = { 0,0,255 };

	CImg<unsigned char> visu(COLUMNS,ROWS,1,3,255);
	CImgDisplay draw_disp(visu,"Cities");

	//draw cities
	for(int i=0;i<nodes;++i)
	{
		visu.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
	}

	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = optTour[i];
		nextNode = optTour[(i+1)%nodes];
		visu.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			red,1,~0U);
	}

	visu.display(draw_disp);

	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
		
	//visu.draw_graph(image.get_crop(0,y,0,2,image.width()-1,y,0,2),blue,1,0,256,0).display(draw_disp); 
}
void displayCurrentHull(CImg<unsigned char>* visu,const unsigned char* color){
	//for debugging purposes.
	for(int i=0;i<nodes;++i){
		if(mask[i]>=0)
		{
			(*visu).draw_line(
			xtoc(x[i]),ytor(y[i]),
			xtoc(x[mask[i]]),ytor(y[mask[i]]),
			color,1,~0U);
		}
	}
}
void drawConvexHull()
{
	fillConstants();


	const unsigned char black[] = {0,0,0};
	const unsigned char red[] = { 255,0,0 };
	const unsigned char green[] = { 0,255,0 };
	//const unsigned char blue[] = { 0,0,255 };

	CImg<unsigned char> visu(COLUMNS,ROWS,1,3,255);
	CImgDisplay draw_disp(visu,"Cities");

	//draw cities
	for(int i=0;i<nodes;++i)
	{
		visu.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
	}

	drawCity(45,&visu,green);
	drawCity(35,&visu,green);
	drawCity(48,&visu,green);

	drawCity(63,&visu,green);



	//draw convex hull
	cout << "Checkpoint 1" << "\n";
	convexHullAlgorithm();
	cout << "Checkpoint 2" << "\n";
	displayCurrentHull(&visu,red);

/*


	int i=0;
	while(mask[i]<0)
	{
		++i;
	}
	cout << "Checkpoint 3: " << i << "\n";
	int firstNode = i;
	int currNode = i;
	int nextNode = mask[currNode];
		cout << "Current Node: " << currNode << "\n";
		cout << "Next Node: " << nextNode << "\n";
	while(nextNode!=firstNode && nextNode>=0)
	{

		visu.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			red,1,~0U);
		currNode = nextNode;
		nextNode = mask[nextNode];
	}
*/
	visu.display(draw_disp);

	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}

void drawCity(int n,CImg<unsigned char>* visu,const unsigned char* color){
	(*visu).draw_circle(xtoc(x[n]),ytor(y[n]),CITYRADIUS,color,1,~0U);
}














coordType getMaxX()
{
	coordType max=x[0];
	for(int i=1;i<nodes;++i)
	{
		if(x[i]>max)
		{
			max=x[i];
		}
	}
	return max;
}
coordType getMinX()
{
	coordType min=x[0];
	for(int i=1;i<nodes;++i)
	{
		if(x[i]<min)
		{
			min=x[i];
		}
	}
	return min;
}
coordType getMinY()
{
	coordType min=y[0];
	for(int i=1;i<nodes;++i)
	{
		if(y[i]<min)
		{
			min=y[i];
		}
	}
	return min;
}
coordType getMaxY()
{
	coordType max=y[0];
	for(int i=1;i<nodes;++i)
	{
		if(y[i]>max)
		{
			max=y[i];
		}
	}
	return max;
}