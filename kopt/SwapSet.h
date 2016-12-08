#pragma once


// Each consecutive pair represents segments
//  (in any order). The indices are relative to an original
//  arrangement of segments [0, 2 * K).


template <std::size_t K>
struct SwapSet
{
    static constexpr std::size_t Length = 2 * K;
    int c[Length];  // relative city index.
};



