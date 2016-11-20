#include "Reader.h"


Reader::Reader(std::string filename) : m_filename(std::move(filename))
{
    readCities();
}


int Reader::getCityCount() const
{
  std::ifstream myfile(m_filename.c_str());
  if(myfile.is_open())
  {
    std::string line;
    while(getline(myfile,line))
    {
      std::vector<std::string> tokens = spaceTokens(line);
      int match = tokens[0] == "DIMENSION";
      if(tokens.size() < 3) match = tokens[0] == "DIMENSION:";
      if(match)
      {
        int cityCount = atoi(tokens[2].c_str());
        std::cout << "Dimension Read: " << cityCount << std::endl;
        return cityCount;
      }
    }
  }
  else std::cerr << "Unable to read city count." << std::endl;
  return 0;
}


void Reader::readCities()
{
  m_cities.clear();
  std::ifstream myfile(m_filename.c_str());
  if(myfile.is_open())
  {
    std::string line;
    while(getline(myfile,line))
    {
      if(line == "NODE_COORD_SECTION") break;
    }
    while(getline(myfile,line))
    {
      std::vector<std::string> tokens = spaceTokens(line);
      if(tokens.size() >= 3)
      {
        const double x = atof(tokens[1].c_str());
        const double y = atof(tokens[2].c_str());
        m_cities.push_back(City{x, y});
      }
    }
  }
  else std::cerr << "Unable to read city coordinates." << std::endl;
}


const std::vector<City>& Reader::getCities() const
{
  return m_cities;
}

std::vector<std::string> Reader::spaceTokens(std::string line) const
{
  std::vector<std::string> tokens;
  std::string buffer;
  std::stringstream ss(line);
  while(ss >> buffer) tokens.push_back(buffer);
  return tokens;
}
