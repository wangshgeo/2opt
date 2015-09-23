#pragma once

// C library headers
#include <cstdlib>

// C++ library headers
#include <fstream>
#include <string>
#include <algorithm>

// Project-specific headers
#include "types.h"



int getCityCount(char* filename);


void fill_coordinates_2D(const char* filename, dtype* x, dtype* y, const int n);


void shuffle_cities(dtype* x, dtype* y, const int n);