#pragma once


// Represents new pairing of set of a segments.
//  The cities of K segments are referred to by relative IDs
//  [0, 2 * K). For each segment k, indices 2 * k and 2 * k + 1
//  refer to segment k's city IDs.


#include <array>


template <std::size_t K>
using SwapSet = std::array<int, 2 * K>;


