#include "include/solution.h"

void Solution::readTour(char*tourFile)
{
  string line;
  ifstream file (tourFile);
  if (file.is_open())
  {
    //before coordinates (header)
    while ( getline (file,line) )
    {
      if(line=="TOUR_SECTION") break;  
    }
    //now onto tour
    int i=0;
    while ( getline (file,line) )
    {
      if(line!="-1" && line!="EOF" && line!="")
      {
        vector<string> tokens = spaceTokens(line);
        if(tokens.size()>1)
        {
			for(int j=i;j<(int)tokens.size();++j)
			{
				tour.push_back(atoi(tokens.at(j-i).c_str()) - 1);
			}
        }
        else
        {
        	tour.push_back(atoi(line.c_str()) - 1);
        }
        ++i;
      }
    }
    file.close();
  }
  else
  {
    cout << "Unable to open file";
  }
}
vector<string> Solution::spaceTokens(string line) const
{
  vector<string> tokens; // Create vector to hold our words
  string buffer;
  stringstream ss(line); // Insert the string into a stream
  while (ss >> buffer)
    tokens.push_back(buffer);
  return tokens;
}
void Solution::updateTourLength(const Instance& instance)
{
	coordType cost=0;

	tourContainer::iterator it=tour.begin();
	int prev=*it,current;
	++it;
	for(;it!=tour.end();++it)
	{	
		current=*it;
		cost+=instance.getDistance(prev,current);
		prev=current;
	}
	cost+=instance.getDistance(tour.front(),tour.back());
	
	tour_length=cost;
}
coordType Solution::calculateTourLength(const Instance& instance, tourContainer cur_tour)
{
	coordType cost=0;
	tourContainer::iterator it=cur_tour.begin();
	int prev=*it,current;
	++it;
	for(;it!=cur_tour.end();++it)
	{	
		current=*it;
		cost+=instance.getDistance(prev,current);
		prev=current;
	}
	cost+=instance.getDistance(cur_tour.front(),cur_tour.back());
	return cost;
}
void Solution::tourFromOrder(int*order,const Instance& instance)
{
	int city;
  for(int i=0;i<instance.getCityCount();++i)
  {
  	city=-1;
  	for(int j=0;j<instance.getCityCount();++j)
  	{
  		if(i==order[j])
  		{
  			city=j;
  		}
  	}
  	if(city<0)
  	{
  		cout << "Error in constructing tour from order array." << "\n";
  	}
    tour.push_back(city);
  }
}
Solution::Solution(){
  this->setTourLength(0);
}
void Solution::copySolution(Solution* s)
{
  this->setTourLength(s->getTourLength());
  this->tour.clear();
  for (std::list<int>::iterator it = s->tour.begin(); it != s->tour.end(); it++)
  {
    this->tour.push_back(*it);
  }
}
string Solution::tourToString()
{
  ostringstream answ;
  for (std::list<int>::iterator it = this->tour.begin(); it != this->tour.end(); it++)
  {
    answ << *it << " ";
  }
  return answ.str();
}
string Solution::tourToString(tourContainer new_tour)
{
  ostringstream answ;
  for (std::list<int>::iterator it = new_tour.begin(); it != new_tour.end(); it++)
  {
    answ << *it << " ";
  }
  return answ.str();
}
void Solution::addToTour(int idx, float value)
{
  this->tour.push_back(idx);
  this->setTourLength(this->getTourLength()+value);
}
void Solution::tourFromPath(int*path,const Instance& instance)
{//path is indexed by order; path[order] = city id
  for(int i=0;i<instance.getCityCount();++i)
  {
    tour.push_back(path[i]);
  }
  updateTourLength(instance);
}
