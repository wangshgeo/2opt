#pragma once

#include <array>
#include <iostream>
#include <vector>

#include "Set.h"


template <std::size_t Segments>
class Pattern
{
public:
    Pattern();
    void printSets() const;
private:
    static constexpr std::size_t Points = 2 * Segments;
    using CanVec = std::vector<std::size_t>;
    using SegVec = std::vector<Segment>;

    std::vector<SegVec> m_sets;

    void pairOff(std::size_t city, CanVec, SegVec);
    decltype(CanVec()) valid(std::size_t city, CanVec);
    decltype(CanVec()) oneOut(std::size_t city, CanVec);
    inline std::size_t next(std::size_t c) { return (c + 1) % Points; }
    inline std::size_t prev(std::size_t c) { return (c + Points - 1) % Points; }
};


template <std::size_t Segments>
Pattern<Segments>::Pattern()
{
    CanVec can;
    can.resize(Points - 1);
    std::size_t i = 0;
    for(auto& c : can)
    {
        c = ++i;
    }
    pairOff(0, can, SegVec());
}

template <std::size_t Segments>
void Pattern<Segments>::printSets() const
{
    for(auto set : m_sets)
    {
        for(auto c : set)
        {
            std::cout << c.a << " " << c.b << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Swap sets: " << m_sets.size() << "\n";
}

template <std::size_t Segments>
void Pattern<Segments>::pairOff(std::size_t city,
    CanVec candidates, SegVec set)
{
    // Recursive end condition.
    if (set.size() == Segments)
    {
        if (candidates.size() != 0)
        {
            std::cout << "Unexpectedly have a full"
                " set, but candidates remain."
                << std::endl;
        }
        m_sets.push_back(set);
        return;
    }
    // Attempt to pair and pass off for more pairings.
    auto filtered = valid(city, candidates);
    if(filtered.size() > 0)
    {
        for(auto f : filtered)
        {
            auto newSet = set;
            newSet.push_back(Segment(city, f));
            auto newCandidates = oneOut(f, candidates);
            // Pick arbitrary new city.
            std::size_t newCity = 0;
            if (newCandidates.size() > 0)
            {
                newCity = newCandidates.back();
                newCandidates.pop_back();
            }
            pairOff(newCity, newCandidates, newSet);
        }
    }
}

template <std::size_t Segments>
typename Pattern<Segments>::CanVec Pattern<Segments>::oneOut(std::size_t city, CanVec orig)
{
    CanVec ret;
    for(auto c : orig)
    {
        if (c != city)
        {
            ret.push_back(c);
        }
    }
    return ret;
}


template <std::size_t Segments>
typename Pattern<Segments>::CanVec Pattern<Segments>::valid(std::size_t city, CanVec candidates)
{
    CanVec v;
    for(auto c : candidates)
    {
        if (c != city and c != next(city) and c != prev(city))
        {
            v.push_back(c);
        }
    }
    return v;
}



