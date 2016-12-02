#pragma once

#include "Segment.h"

template <int S>
struct Set
{
    int i = 0;
    Segment s[S];

    inline void push(Segment s_);
    inline bool full() const;
};


inline void Set::push(Segment s_)
{
    if(i < S)
    {
        s[i] = s_;
        ++i;
    }
}


bool Set::full() const
{
    return i == S;
}


