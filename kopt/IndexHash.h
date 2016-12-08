#pragma once

#include <vector>

class IndexHash
{
public:
    IndexHash(const int maxIndex);
    // This converts the pairing of i and j to a unique index
    //  that has a contiguous range of values.
    inline int hash(const int i, const int j) const;
private:
    std::vector<int> m_starts; // cityId -> pair hash start

    // Returns the starting serialized index, given i.
    inline int start(const int i) const;
};


int IndexHash::hash(const int i, const int j) const
{
  const int lower = (i < j) ? i : j;
  const int higher = (i > j) ? i : j;
  return m_starts[higher] + lower;
}
