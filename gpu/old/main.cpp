//ROBERT LEE

#include "include/main.h"
#include "include/input.h"
#include "include/drawing.h"

coordType *x,*y; 
int nodes;
int *optTour,*mask;
coordType optScore;
int **distanceTable;

int main () {

  string allCases = "ALL_tsp";
  string caseName="eil101";
  string extension=".tsp";  
  string path=allCases+"/"+caseName+"/"+caseName+extension;
  //cout << "File name: " << path << "\n";
  readTSP(path);
  
  extension=".opt.tour";  
  path=allCases+"/"+caseName+"/"+caseName+extension;
  //cout << "File name: " << path << "\n"; 
  readOptTour(path);
  //for(int i=0;i<nodes;++i){ cout << optTour[i] << "\n"; }

  optScore = tourCost(optTour,nodes);
  cout.precision(9);
  cout << "Optimal Score: " << optScore << "\n";

  drawCities();
  drawConvexHull();

  //displayCoordinates();
  deallocateArrays();
  return 0;
}

void displayCoordinates(){
  cout << "Total nodes: " << nodes << "\n";
  for(int i=0;i<nodes;++i){
    cout << i << ": " << x[i] << " " << y[i] << "\n";
  } 
}

coordType nodeDistance(int n1,int n2){
  coordType dx = (coordType)(x[n1]-x[n2]);
  coordType dy = (coordType)(y[n1]-y[n2]);
  coordType dd = (coordType)sqrt(dx*dx+dy*dy);
  return round(dd);
}
coordType haversine(int n1,int n2){
  /*
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2 
  c = 2 * atan2( sqrt(a), sqrt(1-a) ) 
  d = R * c (where R is the radius of the Earth)
  */
  return 0;
}
coordType tourCost(int *tour, int nt){
  //Goes from node 0 to nt-1, then back to 0.
  coordType sum=0;
  int n1,n2;
  n1=tour[0];
  for(int i=1;i<nt;++i){
    n2=tour[i];
    sum+=nodeDistance(n1,n2);
    n1=n2;
  }
  sum+=nodeDistance(tour[nt-1],tour[0]);
  return sum;
}

void initMask(){
  for(int i=0;i<nodes;++i){
    mask[i]=-1;
  }
}


