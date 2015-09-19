// Writing a unit test using Google C++ testing framework is easy as 1-2-3:

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

// C++ library headers
#include <limits.h>

// GoogleTest library headers
#include "gtest/gtest.h"

// Project-specific headers
#include "quadtree.h"
#include "morton_serial.h"
#include "segment_serial.h"
#include "quadtree_serial.h"

// Step 2. Use the TEST macro to define your tests.
//
// TEST has two parameters: the test case name and the test name.
// After using the macro, you should define your test logic between a
// pair of braces.  You can use a bunch of macros to indicate the
// success or failure of a test.  EXPECT_TRUE and EXPECT_EQ are
// examples of such macros.  For a complete list, see gtest.h.
//
// <TechnicalDetails>
//
// In Google Test, tests are grouped into test cases.  This is how we
// keep test code organized.  You should put logically related tests
// into the same test case.
//
// The test case name and the test name should both be valid C++
// identifiers.  And you should not use underscore (_) in the names.
//
// Google Test guarantees that each test you define is run exactly
// once, but it makes no guarantee on the order the tests are
// executed.  Therefore, you should write your tests in such a way
// that their results don't depend on their order.
//
// </TechnicalDetails>

// Tests factorial of negative numbers.
TEST(FactorialTest, Negative) {
  // This test is named "Negative", and belongs to the "FactorialTest"
  // test case.
  EXPECT_EQ(1, Factorial(-5));
  EXPECT_EQ(1, Factorial(-1));
  EXPECT_GT(Factorial(-10), 0);

  // <TechnicalDetails>
  //
  // EXPECT_EQ(expected, actual) is the same as
  //
  //   EXPECT_TRUE((expected) == (actual))
  //
  // except that it will print both the expected value and the actual
  // value when the assertion fails.  This is very helpful for
  // debugging.  Therefore in this case EXPECT_EQ is preferred.
  //
  // On the other hand, EXPECT_TRUE accepts any Boolean expression,
  // and is thus more general.
  //
  // </TechnicalDetails>
}

// Tests factorial of 0.
TEST(FactorialTest, Zero) {
  EXPECT_EQ(1, Factorial(0));
}

// Tests factorial of positive numbers.
TEST(FactorialTest, Positive) {
  EXPECT_EQ(1, Factorial(1));
  EXPECT_EQ(2, Factorial(2));
  EXPECT_EQ(6, Factorial(3));
  EXPECT_EQ(40320, Factorial(8));
}


// Tests negative input.
TEST(IsPrimeTest, Negative) {
  // This test belongs to the IsPrimeTest test case.

  EXPECT_FALSE(IsPrime(-1));
  EXPECT_FALSE(IsPrime(-2));
  EXPECT_FALSE(IsPrime(INT_MIN));
}

// Tests some trivial cases.
TEST(IsPrimeTest, Trivial) {
  EXPECT_FALSE(IsPrime(0));
  EXPECT_FALSE(IsPrime(1));
  EXPECT_TRUE(IsPrime(2));
  EXPECT_TRUE(IsPrime(3));
}

// Tests positive input.
TEST(IsPrimeTest, Positive) {
  EXPECT_FALSE(IsPrime(4));
  EXPECT_TRUE(IsPrime(5));
  EXPECT_FALSE(IsPrime(6));
  EXPECT_TRUE(IsPrime(23));
}




// Tests basic cases.
TEST(InterleaveIntsTest, Basic)
{
  btype a = (btype)12; // 1100
  btype b = (btype)3; // 0011
  EXPECT_EQ((mtype)165, interleave_ints(a,b)); // 10100101
}

// Tests basic cases.
TEST(SegmentTest, ComputeSegmentLengths)
{
  dtype x[3] = {0, 1, 2};
  dtype y[3] = {0, 0, 0};
  cost_t segments[3] = {-1,-1,-1};
  compute_segment_lengths(x,y,3,segments);
  cost_t correct[3] = {1, 1, 2};
  for( int i = 0; i < 3; ++i )
  {
    EXPECT_EQ(correct[i], segments[i]);
  }
}

// Tests basic cases.
TEST(SegmentTest, GetLevelMsb)
{
  int level = 5;
  mtype quadrant = (mtype)3;
  mtype key = quadrant << (2*(MAX_LEVEL - level));

  EXPECT_EQ(get_level_msb(key,level), quadrant);
  quadrant = (mtype)2;
  key = quadrant << (2*(MAX_LEVEL - level));
  EXPECT_EQ(get_level_msb(key,level), quadrant);
  quadrant = (mtype)1;
  key = quadrant << (2*(MAX_LEVEL - level));
  EXPECT_EQ(get_level_msb(key,level), quadrant);
  quadrant = (mtype)0;
  key = quadrant << (2*(MAX_LEVEL - level));
  EXPECT_EQ(get_level_msb(key,level), quadrant);
}

// Tests basic cases, 3 cities.
TEST(SegmentTest, InsertSegment3)
{
  dtype x[3] = {0.25, 0.25, 0.75};
  dtype y[3] = {0.25, 0.75, 0.25};
  morton_key_type* point_morton_pairs = new morton_key_type[3];
  make_morton_keys_serial(point_morton_pairs, x, y,
    3, // Number of cities
    0, // Min x
    1, // Max x
    0, // Min y
    1 // Max y
  );
  Node* tree = construct_quadtree_serial(
    NULL, // We are not inputing a head, since we are retrieving the head.
    -1, // The head is not a child, so its child index is -1.
    point_morton_pairs,
    3, // Number of cities. 
    0, // Current level of root is 0.
    x, 
    y
  );

  // Check presence of children
  EXPECT_TRUE(tree->getChild(3) == NULL);
  EXPECT_FALSE(tree->getChild(2) == NULL);
  EXPECT_FALSE(tree->getChild(1) == NULL);
  EXPECT_FALSE(tree->getChild(0) == NULL);

  // Check point count
  EXPECT_TRUE(tree->getP() == 3);

  // Diameter check
  dtype x_mean = 1.25 / 3.0;
  dtype y_mean = 1.25 / 3.0;
  dtype dx = x_mean - 0.25;
  dtype dy = y_mean - 0.75;
  dtype distance = sqrt(dx*dx + dy*dy);
  EXPECT_EQ(2*distance, tree->getDiameter());

  // center of mass check
  dtype* center_of_mass = tree->getCenterOfMass();
  EXPECT_EQ(center_of_mass[0], x_mean);
  EXPECT_EQ(center_of_mass[1], y_mean);

  // check if points are stored in root
  leaf_container* points = tree->getPoints();
  EXPECT_EQ(0, points->size());

  // Check children quadrants
  EXPECT_EQ(tree->getChild(0)->getLevelIndex(), 0);
  EXPECT_EQ(tree->getChild(1)->getLevelIndex(), 1);
  EXPECT_EQ(tree->getChild(2)->getLevelIndex(), 2);

  // finally, the segments test part!!!
  mtype point_morton_keys[3] = {0,0,0};
  ordered_point_morton_keys(point_morton_pairs, point_morton_keys, 3);
  insert_segments(tree, point_morton_keys, 3);

  // Check number of segments contained
  EXPECT_EQ(tree->getS(), 3);
  EXPECT_EQ(tree->getChild(0)->getS(), 0);
  EXPECT_EQ(tree->getChild(1)->getS(), 0);
  EXPECT_EQ(tree->getChild(2)->getS(), 0);

  // Check number of leaf segments
  leaf_container* segments = tree->getSegments();
  EXPECT_EQ(segments->size(), 3);


  delete[] point_morton_pairs;
  destroy_quadtree_serial(tree);
}

// Tests basic cases, 4 cities.
TEST(SegmentTest, InsertSegment4)
{
  dtype x[4] = {0.125, 0.125, 0.375, 0.4375};
  dtype y[4] = {0.125, 0.625, 0.875, 0.875};
  morton_key_type* point_morton_pairs = new morton_key_type[4];
  make_morton_keys_serial(point_morton_pairs, x, y,
    4, // Number of cities
    0, // Min x
    1, // Max x
    0, // Min y
    1 // Max y
  );
  Node* tree = construct_quadtree_serial(
    NULL, // We are not inputing a head, since we are retrieving the head.
    -1, // The head is not a child, so its child index is -1.
    point_morton_pairs,
    4, // Number of cities. 
    0, // Current level of root is 0.
    x, 
    y
  );

  // check tree structure
  EXPECT_TRUE(tree->getChild(1)->getChild(3)->getChild(0) == NULL);
  EXPECT_TRUE(tree->getChild(1)->getChild(3)->getChild(1) == NULL);
  EXPECT_TRUE(tree->getChild(1)->getChild(3)->getChild(2) == NULL);
  EXPECT_TRUE(tree->getChild(1)->getChild(3)->getChild(3) == NULL);

  // finally, the segments test part!!!
  mtype point_morton_keys[4] = {0,0,0,0};
  ordered_point_morton_keys(point_morton_pairs, point_morton_keys, 4);
  // insert_segment(tree, point_morton_keys, 0,1);
  // insert_segment(tree, point_morton_keys, 1,2);
  // insert_segment(tree, point_morton_keys, 2,3);
  // insert_segment(tree, point_morton_keys, 3,0);
  insert_segments(tree, point_morton_keys, 4);

  // Check number of segments exclusive to nodes.
  EXPECT_EQ(tree->getS(), 2);
  EXPECT_EQ(tree->getChild(1)->getS(), 1);
  EXPECT_EQ(tree->getChild(1)->getChild(3)->getS(), 1);

  // Check number of total segments under nodes.
  EXPECT_EQ(tree->getTotalS(), 4);
  EXPECT_EQ(tree->getChild(1)->getTotalS(), 2);
  EXPECT_EQ(tree->getChild(1)->getChild(3)->getS(), 1);

  // Check number of leaf segments
  leaf_container* segments = tree->getSegments();
  EXPECT_EQ(segments->size(), 2);
  EXPECT_EQ((*segments)[0], 0);
  EXPECT_EQ((*segments)[1], 3);
  segments = tree->getChild(1)->getSegments();
  EXPECT_EQ(segments->size(), 1);
  EXPECT_EQ((*segments)[0], 1);
  segments = tree->getChild(1)->getChild(3)->getSegments();
  EXPECT_EQ(segments->size(), 1);
  EXPECT_EQ((*segments)[0], 2);


  delete[] point_morton_pairs;
  destroy_quadtree_serial(tree);
}

// Step 3. Call RUN_ALL_TESTS() in main().
//
// We do this by linking in src/gtest_main.cc file, which consists of
// a main() function which calls RUN_ALL_TESTS() for us.
//
// This runs all the tests you've defined, prints the result, and
// returns 0 if successful, or 1 otherwise.
//
// Did you notice that we didn't register the tests?  The
// RUN_ALL_TESTS() macro magically knows about all the tests we
// defined.  Isn't this convenient?
