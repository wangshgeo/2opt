#pragma once


// Represents new pairing of set of a segments.
//  The index of the SwapSet refers to the first city of
//  each segment.
//  The value of the SwapSet refers to the second city of
//  each segment.
// The cities of K segments are referred to by relative IDs
//  [0, 2 * K).


#include <array>


template <std::size_t K>
using SwapSet = std::array<int, K>;


