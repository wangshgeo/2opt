#include "Reader.h"


Reader::Reader(std::string filename) : m_filename(std::move(filename))
{
    readCities();
}


void Reader::readCities()
{
    m_cities.clear();
    std::ifstream myfile(m_filename.c_str());
    if(myfile.is_open())
    {
        const std::string Tag("NODE_COORD_SECTION");
        std::string line;
        while(getline(myfile,line))
        {
            if(line.find(Tag) != std::string::npos)
            {
                break;
            }
        }
        while(getline(myfile,line))
        {
            std::vector<std::string> tokens = spaceTokens(line);
            if(tokens.size() >= 2)
            {
                const double x
                    = atof((*(tokens.end()-2)).c_str());
                const double y = atof(tokens.back().c_str());
                m_cities.push_back(City{x, y});
            }
        }
    }
    else
    {
        std::cerr << "Unable to read city coordinates."
            << std::endl;
    }
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
    while(ss >> buffer)
    {
        tokens.push_back(buffer);
    }
    return tokens;
}


CostFunction Reader::getCostFunction() const
{
    std::ifstream myfile(m_filename.c_str());
    if(myfile.is_open())
    {
        std::string line;
        while(getline(myfile,line))
        {
            const std::string Tag("EDGE_WEIGHT_TYPE");
            if(line.find(Tag) != std::string::npos)
            {
                if(line.find("GEO", Tag.size()))
                {
                    return CostFunction::GEO;
                }
                if(line.find("EUC", Tag.size()))
                {
                    return CostFunction::EUC;
                }
                return CostFunction::NONE;
            }
        }
    }
    else
    {
        std::cerr << "Unable to read cost function type."
            << std::endl;
    }
    return CostFunction::NONE;
}


int Reader::getCityCount() const
{
    std::ifstream myfile(m_filename.c_str());
    if(myfile.is_open())
    {
        const std::string Tag("DIMENSION");
        std::string line;
        while(getline(myfile,line))
        {
            if(line.find(Tag) != std::string::npos)
            {
                const std::vector<std::string> tokens
                    = spaceTokens(line);
                const int cityCount
                    = atoi(tokens.back().c_str());
                std::cout << "Dimension Read: "
                    << cityCount << std::endl;
                return cityCount;
            }
        }
    }
    else
    {
        std::cerr << "Unable to read city count." << std::endl;
    }
    return 0;
}



