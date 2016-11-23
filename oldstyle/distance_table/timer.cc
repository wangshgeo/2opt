#include <assert.h>
#include <stdlib.h>
#include "timer.hh"

stopwatch_t 
stopwatch_start(){
  struct timeval* start;

  start = (struct timeval*) malloc(sizeof(struct timeval));

  gettimeofday(start, NULL);

  return (stopwatch_t) start;
}

long double
stopwatch_stop(stopwatch_t watch){
  struct timeval stop;
  struct timeval *start = (struct timeval *) watch;

  gettimeofday(&stop, NULL);

  long double ans = (long double)(stop.tv_sec - start->tv_sec) + (long double)(stop.tv_usec - start->tv_usec)*1e-6;

  free(watch);

  return ans;
}


