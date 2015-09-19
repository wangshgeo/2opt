#pragma once

typedef float dtype;
typedef long int cost_t;

typedef unsigned int btype;
typedef unsigned long int mtype;

#define MAX_LEVEL 21 //maximum level / depth of the quadtree. Leave at least one bit for flags.

#define MORTON_FUDGE_FACTOR 0.05 //multiplier to maximum ranges so that extreme points dont become <0 or >1 due to finite precision. 
