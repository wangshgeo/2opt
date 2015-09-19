#include "include/types.h"
#include "include/geometric.h"
#include "include/main.h"

extern coordType *x,*y;
extern int nodes;
extern int *optTour,*mask;
extern coordType optScore;

int findMinXNode()
{
	int minNode=0;
	for(int i=1;i<nodes;++i)
	{
		if(x[i]<x[minNode])
		{
			minNode=i;
		}
	}
	return minNode;
}
coordType getAngle(int n1,int n2)
{
	//gets angle formed between the horizontal and the line made up by n1 and n2.
	//n1 is the origin.
	coordType dx = x[n2]-x[n1];
	coordType dy = y[n2]-y[n1];
	return atan2(dy,dx); 
}
coordType spanAngle(coordType dx1,coordType dy1,coordType dx2,coordType dy2)
{
	coordType a1 = atan2(dy1,dx1);
	coordType a2 = atan2(dy2,dx2);
	/*
	if(a1 > MYPI)
	{
		a1 = -((coordType)MYPI2-a1);
	}
	if(a2 > MYPI)
	{
		a2 = -((coordType)MYPI2-a2);
	}
	*/
	coordType span =fabs(a1-a2);
	if(span > MYPI){
		span=MYPI2-span;
	}
	return span;
}
int findMaxSpanNode(int pivot,int prevPivot,coordType dx,coordType dy)
{
	//dx,dy goes from pivot to node.
	coordType dx_,dy_;
	int maxNode=0;
	coordType angle,maxSpan=0;
	for(int i=0;i<nodes;++i)
	{
		if(i!=prevPivot && i!=pivot)
		{
			dx_=x[i]-x[pivot];
			dy_=y[i]-y[pivot];
			angle = spanAngle(dx,dy,dx_,dy_);
			if(angle>maxSpan)
			{
				maxSpan=angle;
				maxNode=i;
			}
		}
	}
	//cout << "Max Span Node: " << maxNode << ", Span Angle: " << maxSpan << "\n";
	return maxNode;
}
void findConvexHull()
{
	//fills the mask array with the next node value if the point is part of the convex hull.
	//cout << "Check Point findConvexHull";
	//find first convex nodes.
	int first = findMinXNode();
	coordType angle,minAngle,maxAngle;
	int minNode=0,maxNode=0;
	int i=0;
	if(i==first)
	{
		++i;
	}
	maxAngle=getAngle(first,i);
	minAngle=maxAngle;

	for(i=0;i<nodes;++i)
	{
		if(i!=first)
		{
			angle = getAngle(first,i);
			if(angle<minAngle)
			{
				minAngle=angle;
				minNode=i;
			}
			if(angle>maxAngle)
			{
				maxAngle=angle;
				maxNode=i;
			}
		}
	}
	mask[first] = maxNode;
	mask[minNode] = first;
	cout << minNode << ", " << first << ", " << maxNode << "\n";
	
	int currMax,currPivot,prevPivot;
	coordType dx,dy;
	currPivot=maxNode;
	prevPivot=first;
	dx=x[prevPivot]-x[currPivot];
	dy=y[prevPivot]-y[currPivot];
	//i=0;
	//while(i<8){
	//	++i;
	while(currPivot!=minNode)
	{
		//cout << "atan2(dy,dx): " << atan2(dy,dx) << "\n";
		//cout << "\nCurrent Pivot: " << currPivot << "\n";
		//cout << "dx,dy: " << dx << "," << dy << "\n";
		//cout << "atan2(dy,dx): " << atan2(dy,dx) << "\n";
		currMax = findMaxSpanNode(currPivot,prevPivot,dx,dy);
		//cout << "Current Max: " << currMax << "\n";
		mask[currPivot] = currMax;
		//Advance
		prevPivot = currPivot;
		currPivot = currMax;
		dx = x[prevPivot]-x[currPivot];
		dy = y[prevPivot]-y[currPivot];
	}
	//cout << "atan2(dy,dx): " << atan2(0,-1) << "\n";
	
}

void convexHullAlgorithm(){
	//cout << "Check Point convexHullAlgorithm" << "\n";
	initMask();
	findConvexHull();
}