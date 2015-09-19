#!/bin/bash

TSPFILE="ALL_tsp/usa13509.tsp"
# ./opt2 $TSPFILE
nvprof ./opt2 $TSPFILE
