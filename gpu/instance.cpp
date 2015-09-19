#include "include/instance.h"

#define PI 3.141592
#define RRR 6378.388

void Instance::instanceFromFile(char*coordinateFile)
{
	nb_cities = readCityCount(coordinateFile);
	//cout << nb_cities << "\n";
	coordType x[nb_cities];
	coordType y[nb_cities];
	readCityCoordinates(coordinateFile,x,y);
	int df = readEdgeWeightType(coordinateFile);
	table = new coordType*[nb_cities];
	angles = new coordType*[nb_cities];
	for(int i=0;i<nb_cities;++i)
	{
		table[i]=new coordType[nb_cities];
		angles[i]=new coordType[nb_cities];
	}
	coords = new COORD[nb_cities];
	for(int i=0;i<nb_cities;++i)
	{
		coords[i].x = x[i];
		coords[i].y = y[i];
	}
	constructDistanceTable(x,y,df);
	constructAngles(x,y);
}

int Instance::readCityCount(char*filepath) const
{
  int citycount=-1;
  ifstream myfile (filepath); 
  if (myfile.is_open())
  {
    string line;
    while ( getline (myfile,line) )
    {
      vector<string> tokens = spaceTokens(line);
      //if(tokens[0]=="DIMENSION:")
      int match = tokens[0]=="DIMENSION";
      if(tokens.size()<3)
      {
      	match = tokens[0]=="DIMENSION:";
      }
      if(match)
      {
        //citycount=atoi(tokens[1].c_str());
        citycount=atoi(tokens[2].c_str());
        fprintf(stderr,"Dimension Read: %d\n",citycount);
        //cout << "Dimension Read: " << citycount << "\n";
        break;
      }
    }
  }
  else
  {
        fprintf(stderr,"Unable to open specified file while reading city count.\n");
   		//cout << "Unable to open specified file while reading city count." << "\n";
  }
	myfile.close();
  return citycount;
}
void Instance::readCityCoordinates(char*filepath,coordType*x,coordType*y) const
{
  ifstream myfile (filepath); 
  if (myfile.is_open())
  {
    string line;
    //skip to the node coord section
    while ( getline (myfile,line) )
    {
      if(line=="NODE_COORD_SECTION")
      {
        break;
      }
    }
    while ( getline (myfile,line) )
    {
      vector<string> tokens = spaceTokens(line);
      if(tokens.size()>=3){
        int node = atoi(tokens[0].c_str())-1;
        x[node] = atof(tokens[1].c_str());
        y[node] = atof(tokens[2].c_str());
        //cout << x[node] << "," << y[node] << "\n";
      }
    }
  }
  else
  {
  	fprintf(stderr,"Unable to open specified file while reading city coordinates.\n");
  	//cout << "Unable to open specified file while reading city coordinates." << "\n";
  }
	myfile.close();
}
int Instance::readEdgeWeightType(char*filepath) const
{
	//Returns a function pointer to the correct distance function.
	string line;
	string edgeWeightType;
	ifstream myfile (filepath);
	int df=1; 
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			vector<string> tokens = spaceTokens(line);
			//if(tokens[0]=="EDGE_WEIGHT_TYPE :")
		      int match = tokens[0]=="EDGE_WEIGHT_TYPE";
		      if(tokens.size()<3)
		      {
		      	match = tokens[0]=="EDGE_WEIGHT_TYPE:";
		      }
			if(tokens[0]=="EDGE_WEIGHT_TYPE")
			{
	      		//edgeWeightType=tokens[1];
	      		edgeWeightType=tokens[2];
	      		if(edgeWeightType=="GEO")
	      		{
	        		df=0;//Instance::distanceGEO;
	      		}
	      		if(edgeWeightType=="EUC_2D")
	      		{
	        		df=1;//Instance::distanceEUC_2D;
	      		}
      			//cout << "Edge Weight Type Read: " << edgeWeightType << "\n";
      			break;
    		}
  		}
  	}
  	else
  	{
  		fprintf(stderr,"Unable to open specified file while reading edge weight type.\n");
  		//cout << "Unable to open specified file while reading edge weight type." << "\n";
  	}
	myfile.close();
  	return df;
}
vector<string> Instance::spaceTokens(string line) const
{
  vector<string> tokens; // Create vector to hold our words
  string buffer;
  stringstream ss(line); // Insert the string into a stream
  while (ss >> buffer)
    tokens.push_back(buffer);
  return tokens;
}
void Instance::deallocateTable()
{
	for(int i=0;i<nb_cities;++i)
	{
		delete[] table[i];	
		delete[] angles[i];		
	}
	delete[] table;
	delete[] angles;
}
coordType Instance::coordinateToLongitudeLatitude(coordType coordinate) const
{
	//convert coordinate input to longitude and latitude in radians.
	int deg = (int) coordinate;
	coordType min = coordinate-deg;
	coordType rad = PI*(deg+5.0*min/3.0)/180.0;
	return rad;
}
coordType Instance::distanceGEO(coordType*x,coordType*y,int n1,int n2) const
{
	coordType xlat = coordinateToLongitudeLatitude(x[n1]);
	coordType xlong = coordinateToLongitudeLatitude(y[n1]);
	coordType ylat = coordinateToLongitudeLatitude(x[n2]);
	coordType ylong = coordinateToLongitudeLatitude(y[n2]);
	coordType q1 = cos( xlong - ylong );
	coordType q2 = cos( xlat - ylat );
	coordType q3 = cos( xlat + ylat );
	coordType d12 = (int) ( RRR * acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0);
	return d12;
}
coordType Instance::distanceEUC_2D(coordType*x,coordType*y,int n1,int n2) const
{
	coordType dx = x[n1]-x[n2];
	coordType dy = y[n1]-y[n2];
	coordType dd = sqrt(dx*dx+dy*dy);
	return round(dd);
}
void Instance::constructDistanceTable(coordType*x,coordType*y,int df)
{
	if(df==0)
	{
		for(int i=0;i<nb_cities;++i){
			for(int j=0;j<nb_cities;++j){
				table[i][j]=distanceGEO(x,y,i,j);
			}
		}
	}
	else
	{
		for(int i=0;i<nb_cities;++i){
			for(int j=0;j<nb_cities;++j){
				table[i][j]=distanceEUC_2D(x,y,i,j);
			}
		}
	}
}
void Instance::constructAngles(coordType*x,coordType*y)
{//now construct angles; angles[i][j] means the angle of the vector going from i to j
	//y=0,x>0 is the 0-angle line; positive ccw.
	coordType dx,dy;
	for(int j=0;j<nb_cities;++j)
	{
		for(int i=0;i<nb_cities;++i)
		{
			dx=x[j]-x[i];
			dy=y[j]-y[i];
			angles[i][j] = atan2(dy,dx);
			if(angles[i][j] < 0)
			{
				angles[i][j]+=2*PI;
			}
		}
	}
}