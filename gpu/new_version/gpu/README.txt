Test command:
./tsp ../inputs/usa13509.tsp

Profiling command:
nvprof ./tsp ../inputs/usa13509.tsp

Remote profiling command:
nvprof --output-profile remote_profile --analysis-metrics --kernels kernel ./tsp ../inputs/usa13509.tsp 

Expected fist iteration result for usa13509.tsp:
GPU min diff,i,j,raw k : -1050971.125000,124,12476,27910802