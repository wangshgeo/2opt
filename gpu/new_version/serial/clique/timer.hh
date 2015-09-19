#pragma once

extern "C" 
{

	#include <sys/time.h>

	typedef struct timeval* stopwatch_t;

	stopwatch_t stopwatch_start();

	long double stopwatch_stop(stopwatch_t);

}
