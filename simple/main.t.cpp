#include <cassert>
#include <iostream>
#include "Tester.h"

int main(int argc, char* argv[])
{
  World w;
  Tester t(w);
  // 2 cities
  t.testSerialize(1, 0, 0);
  // 3 cities
  t.testSerialize(2, 0, 1);
  t.testSerialize(2, 1, 2);
  // 4 cities
  t.testSerialize(3, 0, 3);
  t.testSerialize(3, 1, 4);
  t.testSerialize(3, 2, 5);
  // 5 cities
  t.testSerialize(4, 0, 6);
  t.testSerialize(4, 1, 7);
  t.testSerialize(4, 2, 8);
  t.testSerialize(4, 3, 9);
  // 6 cities
  t.testSerialize(5, 0, 10);
  t.testSerialize(5, 1, 11);
  t.testSerialize(5, 2, 12);
  t.testSerialize(5, 3, 13);
  t.testSerialize(5, 4, 14);

  std::cout << "Passed all tests." << std::endl;
  return 0;
}