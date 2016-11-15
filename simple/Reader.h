#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "City.h"

class Reader
{
  public:
    Reader(std::string filename) : m_filename(std::move(filename)) {}
    std::vector<City> getCities() const;
    int getCityCount() const;
  private:
    std::string m_filename;
    std::vector<std::string> spaceTokens(std::string line) const;
};

