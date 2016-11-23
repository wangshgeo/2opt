#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "City.h"
#include "CostFunction.h"


class Reader
{
public:
    Reader(std::string filename);
    const std::vector<City>& getCities() const;
    CostFunction getCostFunction() const;
private:
    std::string m_filename;
    std::vector<City> m_cities;

    void readCities();
    std::vector<std::string> spaceTokens(std::string line) const;
    int getCityCount() const;
};

