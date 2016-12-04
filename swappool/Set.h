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

template <int S>
inline void Set<S>::push(Segment s_)
{
    if(i < S)
    {
        s[i] = s_;
        ++i;
    }
}

template <int S>
bool Set<S>::full() const
{
    return i == S;
}


