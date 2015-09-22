#ifndef INSTANCE_H_
#define INSTANCE_H_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// This represents a TSPLIB-formatted Euclidean TSP instance.
// The instance definition includes city coordinates and the number of cities.
class Instance
{
public:
  Instance(string file_name)
  {
    readCities(file_name);
    x_ = new double[cities_];
    y_ = new double[cities_];
    readCoordinates(file_name);
  }
  int cities() { return cities_; }
  double* x() { return x_; }
  double* y() { return y_; }
  ~Instance()
  {
    delete[] x_;
    delete[] y_;
  }
private:
  int cities_; // number of cities in this TSP instance.
  double* x_; // x-coordinate of cities.
  double* y_; // y-coordinate of cities.
  void readCities(string file_name); // Simply obtains the city count from the 
    //file.
  void readCoordinates(string file_name); // Reads in the coordinates.
};



#endif