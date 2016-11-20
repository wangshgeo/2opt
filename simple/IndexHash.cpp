#include "IndexHash.h"


IndexHash::IndexHash(const int cityCount) : m_starts(cityCount, 0)
{
    for(int i = 0; i < cityCount; ++i)
    {
        m_starts[i] = start(i);
    }
}


int IndexHash::start(const int i) const
{
  return ((i * i) >> 1) - (i >> 1);
}
