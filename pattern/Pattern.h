#pragma once


#include <array>
#include <vector>

#include "Set.h"


template <int K>
class Pattern
{
public:
    template <int C>
    void pair(const int city, std::array<int, C> candidates);
private:
    std::vector<Set<K>> m_sets;
    Set<K> m_set;

    template <int C>
    std::array<int, C-2> valid(const int city, std::array<int, C> candidates);
    template <int C>
    inline int next(const int a) { return (a + 1) % C; }
    template <int C>
    inline int prev(const int a) { return (a + C - 1) % C; }
};


template <int K>
template <int C>
void Pattern<K>::pair(const int city, std::array<int, C> candidates)
{
    if(C == 0)
    {
        m_sets.push_back(m_set);
        return;
    }
    for(
}



template <int K>
template <int C>
std::array<int, C-2> Pattern<K>::valid(const int city, std::array<int, C> candidates)
{
    std::array<int, C-2> v;
    int i = 0;
    for(int k = 0; k < C; ++k)
    {
        if(k != i and k != next(i) and k != prev(i))
        {
            v[i] = k;
            ++i;
        }
    }
}



