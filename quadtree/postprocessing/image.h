#pragma once

#include <stdlib.h>
#include <string>

#include "CImg.h"

using namespace cimg_library;

#define IMAGE_SIZE 800
#define BODY_RADIUS 2





void write_static_image(std::string output_file_name,
  const double*x, const double*y, const int n );

