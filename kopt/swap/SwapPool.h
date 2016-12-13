#pragma once


#include <array>
#include <iostream>
#include <vector>

#include "SimpleSegment.h"


template <std::size_t Segments>
class SwapPool
{
    static constexpr std::size_t Points = 2 * Segments;
    using CanVec = std::vector<std::size_t>;
    using SegVec = std::vector<SimpleSegment>;
public:
    SwapPool();
    std::vector<SegVec>& getPool() const;
    void printSets() const;
    void printCount() const;
private:
    std::vector<SegVec> m_sets;

    void pairOff(std::size_t city, CanVec, SegVec);
    decltype(CanVec()) valid(std::size_t city, CanVec);
    decltype(CanVec()) oneOut(std::size_t city, CanVec);
    inline std::size_t next(std::size_t c) { return (c + 1) % Points; }
    inline std::size_t prev(std::size_t c) { return (c + Points - 1) % Points; }
};


template <std::size_t Segments>
typename std::vector<typename SwapPool<Segments>::SegVec>& SwapPool<Segments>::getPool() const
{
    return m_sets;
}


template <std::size_t Segments>
SwapPool<Segments>::SwapPool()
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
void SwapPool<Segments>::printSets() const
{
    for(auto set : m_sets)
    {
        for(auto c : set)
        {
            std::cout << c[0] << " " << c[1] << " ";
        }
        std::cout << "\n";
    }
    printCount();
}


template <std::size_t Segments>
void SwapPool<Segments>::printCount() const
{
    std::cout << "Swap sets for " << Segments
        << "-Opt: " << m_sets.size() << "\n";
}


template <std::size_t Segments>
void SwapPool<Segments>::pairOff(std::size_t city,
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
            newSet.push_back(SimpleSegment{city, f});
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
typename SwapPool<Segments>::CanVec SwapPool<Segments>::oneOut(std::size_t city, CanVec orig)
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
typename SwapPool<Segments>::CanVec SwapPool<Segments>::valid(std::size_t city, CanVec candidates)
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



