#pragma once


// Represents a set of segments that can exchange cities
//  to create new segments.


#include <array>

#include "Segment.h"


template <std::size_t K>
using SwapSet = std::array<Segment*, K>;


