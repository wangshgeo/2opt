#include "Plotter.h"

//View sizing parameters
int Plotter::toColumn(const double x)
{
	coordType scale = 1.0;
	if(xdomain < yrange)
		scale = xdomain/yrange;
	return (int)(COLUMNDOMAIN*(scale*(x-xmin)/xdomain))+PADDING;
}
int Plotter::toRow(const double y)
{
	coordType scale = 1.0;
	if(yrange < xdomain)
		scale = yrange/xdomain;
	return (int)(ROWRANGE*(scale*(ymax-y)/yrange))+PADDING;
}
void Drawer::drawCities()
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
	}

	//Display
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}
void Drawer::drawTour(int*tour)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
		vv.draw_text(xtoc(x[i]),ytor(y[i]),to_string(i).c_str(),black);
	}
	const unsigned char blue[] = {0,0,255};//{R,G,B}, 0-255
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			blue,1,~0U);
	}
	//Display
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}
void Drawer::saveTour(int*tour,int num)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
		vv.draw_text(xtoc(x[i]),ytor(y[i]),to_string(i).c_str(),black);
	}
	const unsigned char blue[] = {0,0,255};//{R,G,B}, 0-255
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			blue,1,~0U);
	}
	string name = "images/optimal"+to_string(num)+".png";
	vv.save(name.c_str());
	/*
	//Display
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
	*/
}
void Drawer::drawTour2(int*tour,int*path,int np)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
		vv.draw_text(xtoc(x[i]),ytor(y[i]),to_string(i).c_str(),black);
	}
	const unsigned char red[] = {255,0,0};//{R,G,B}, 0-255
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			red,1,~0U);
	}
	const unsigned char green[] = {0,255,0};//{R,G,B}, 0-255
	//draw optimal tour
	for(int i=0;i<np-1;++i)
	{
		currNode = path[i];
		nextNode = path[i+1];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			green,1,~0U);
	}

	//Display
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}
void Drawer::drawSeq(int tour[],int seq[],int seqi)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
		vv.draw_text(xtoc(x[i]),ytor(y[i]),to_string(i).c_str(),black);
	}
	const unsigned char blue[] = {0,0,255};//{R,G,B}, 0-255
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			blue,1,~0U);
	}
	const unsigned char red[] = {255,0,0};//{R,G,B}, 0-255
	const unsigned char green[] = {0,255,0};//{R,G,B}, 0-255
	//draw optimal tour
	for(int i=0;i<seqi*2;++i)
	{
		currNode = tour[seq[i]];
		nextNode = tour[seq[(i+1)%(2*seqi)]];
		if(i & 1)
		{
			vv.draw_line(
				xtoc(x[currNode]),ytor(y[currNode]),
				xtoc(x[nextNode]),ytor(y[nextNode]),
				red,1,~0U);
		}
		else
		{
			vv.draw_line(
				xtoc(x[currNode]),ytor(y[currNode]),
				xtoc(x[nextNode]),ytor(y[nextNode]),
				green,1,~0U);
		}
	}
	//Display
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}
void Drawer::saveSeq(int tour[],int seq[],int seqi,int num,int num2)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char black[] = {0,0,0};//{R,G,B}, 0-255
	//draw cities
	for(int i=0;i<nodes;++i)
	{
		//cout << xtoc(x[i],xmin,xdomain) << ", " << ytor(y[i],ymax,yrange) << "\n";
		vv.draw_circle(xtoc(x[i]),ytor(y[i]),CITYRADIUS,black,1,1);
		vv.draw_text(xtoc(x[i]),ytor(y[i]),to_string(i).c_str(),black);
	}
	const unsigned char blue[] = {0,0,255};//{R,G,B}, 0-255
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			blue,1,~0U);
	}
	const unsigned char red[] = {255,0,0};//{R,G,B}, 0-255
	const unsigned char green[] = {0,255,0};//{R,G,B}, 0-255
	//draw optimal tour
	for(int i=0;i<seqi*2;++i)
	{
		currNode = tour[seq[i]];
		nextNode = tour[seq[(i+1)%(2*seqi)]];
		if(i & 1)
		{
			vv.draw_line(
				xtoc(x[currNode]),ytor(y[currNode]),
				xtoc(x[nextNode]),ytor(y[nextNode]),
				red,1,~0U);
		}
		else
		{
			vv.draw_line(
				xtoc(x[currNode]),ytor(y[currNode]),
				xtoc(x[nextNode]),ytor(y[nextNode]),
				green,1,~0U);
		}
	}
	string name = "images/swap_"+to_string(num)+"_"+to_string(num2)+".png";
	vv.save(name.c_str());
}
void Drawer::drawTourSteps(int*tour)
{
	CImg<unsigned char> vv(COLUMNS,ROWS,1,3,255);
	const unsigned char red[] = {255,0,0};//{R,G,B}, 0-255
	CImgDisplay draw_disp(vv,"The Traveling Salesman");
	//draw optimal tour
	int currNode,nextNode;
	for(int i=0;i<nodes;++i)
	{
		currNode = tour[i];
		nextNode = tour[(i+1)%nodes];
		vv.draw_line(
			xtoc(x[currNode]),ytor(y[currNode]),
			xtoc(x[nextNode]),ytor(y[nextNode]),
			red,1,~0U);
		vv.display(draw_disp);
		draw_disp.wait(100);
	}

	//Display
	vv.display(draw_disp);
	while ( !draw_disp.is_closed() )
	{
		draw_disp.wait();
	}
}
coordType Drawer::getMaxX()
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
coordType Drawer::getMinX()
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
coordType Drawer::getMinY()
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
coordType Drawer::getMaxY()
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
