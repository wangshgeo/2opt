#ifndef __SOLUTION_H__
#define __SOLUTION_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <stdlib.h>
#include <iterator>
using namespace std;

#include "types.hh"
#include "Instance.hh"

typedef list<int> tourContainer;

class Solution
{
private:
        dtype tour_length;
        tourContainer tour;
        double runtime;
        //for reading in files
        void readTour(char*tourFile);
        vector<string> spaceTokens(string line) const;
        //for constructor
        void tourFromOrder(int*order,const Instance& instance);
        void tourFromPath(int*path,const Instance& instance);
public:
        void updateTourLength(const Instance& instance);//From tour, computes and stores tour_length.
        static dtype calculateTourLength(const Instance& instance, tourContainer cur_tour);
        void addToTour(int idx) { tour.push_back(idx); }
        void addToTour(int idx, float value);
        string tourToString();
        static string tourToString(tourContainer new_tour);
        //getters
        int getTourNbCities() const {return tour.size();};
        tourContainer getTour() const {return tour;};
        dtype getTourLength() const { return tour_length; }
        double getRuntime() const { return runtime; }
        //setters
        void setTourLength(dtype value) { tour_length = value; }
        void setTour(tourContainer new_tour) { tour = new_tour; };
        void setRuntime(double rt){ runtime = rt; }
        //constructor
        Solution();
        Solution* clone() const {return new Solution(*this); };
        void copySolution(Solution* s);
        Solution(char*tourFile,const Instance& instance) { readTour(tourFile);updateTourLength(instance); }
        Solution(int*order,const Instance& instance) { tourFromOrder(order,instance);updateTourLength(instance); }
        Solution(int*order,const Instance& instance,double rt) { tourFromOrder(order,instance);updateTourLength(instance);runtime=rt; }
        Solution(int*path,const Instance& instance,double rt,int dummyVariable) { tourFromPath(path,instance);runtime=rt; }
};


#endif
