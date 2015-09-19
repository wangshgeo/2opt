#include "include/types.h"
#include "include/input.h"
#include <math.h>

extern coordType *x,*y;
extern int nodes;
extern int *optTour,*mask;
extern coordType optScore;
extern int **distanceTable;

void allocateArrays(int size){
  nodes=size;
  x=new coordType[size];
  y=new coordType[size];
  optTour= new int[size];
  mask= new int[size];
  distanceTable = new int*[size];
  for(int i=0;i<nodes;++i)
  {
    *distanceTable=new int[size];
  }
}
void deallocateArrays(){
  delete[] x;
  delete[] y;
  delete[] optTour;
  delete[] mask;
  for(int i=0;i<nodes;++i)
  {
    delete[] distanceTable[i];
  }
  delete[] distanceTable;
}

void readOptTour(string path){
  //assumes TSP has already been read.
  //does not record the last (negative) nnumber; cycle is implied.
  string line;
  ifstream file (path.c_str());
  if (file.is_open())
  {
    //before coordinates (header)
    while ( getline (file,line) )
    {
      if(line=="TOUR_SECTION") break;  
    }
    int i=0;
    while ( getline (file,line) )
    {
      if(line!="-1" && line!="EOF")
      {
        optTour[i] = (int)atoi(line.c_str()) - 1;
        ++i;
      }
    }
    file.close();
  }
  else
    cout << "Unable to open file";
}

void readTSP(string path){
  string line;
  ifstream myfile (path.c_str()); 
  if (myfile.is_open())
    {
      //before coordinates (header)
      int dimension = readTSPHeader(&myfile);
      cout << "Dimension: " << dimension << "\n";
      allocateArrays(dimension);
      
      //coordinates
      readTSPCoordinates(&myfile);

      myfile.close();
    }
  else
    cout << "Unable to open file";
}

int readTSPHeader(ifstream *file){
  string line;
  int dimension=-1;
  while ( getline (*file,line) )
  {
    if(line=="NODE_COORD_SECTION") break;     
    vector<string> tokens = spaceTokens(line);
    if(tokens[0]=="DIMENSION") dimension=atoi(tokens[2].c_str());
  }
  return dimension;
}
void readTSPCoordinates(ifstream *file){
  string line;
  while ( getline (*file,line) )
  {
    vector<string> tokens = spaceTokens(line);
    if(tokens.size()>=3){
      int node = atoi(tokens[0].c_str())-1;
      //cout << node << "\n";
      x[node] = atof(tokens[1].c_str());
      y[node] = atof(tokens[2].c_str());
    }
  }
}
vector<string> spaceTokens(string line){
  vector<string> tokens; // Create vector to hold our words
  string buffer;
  stringstream ss(line); // Insert the string into a stream
  while (ss >> buffer)
    tokens.push_back(buffer);
  return tokens;
}

void fillDistanceTable(){
  coordType dx,dy;
  for(int i=0;i<nodes;++i)
  {
    for(int j=0;j<nodes;++i)
    {
      dx=x[i]-x[j];
      dy=y[i]-y[j];
      distanceTable[i][j]=sqrt(dx*dx+dy*dy);
    }
  }
}