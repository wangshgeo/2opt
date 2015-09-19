#pragma once

//New kernel parameters.
#define THREAD_CHUNK 5 //When squared, equals the number of evaluations per thread.
#define BLOCK_DIMENSION 8 //When squared, should be multiple of warp size!
//end.



#define MAX_ITER 100
#define REFRESH_ITER 20 //number of iterations before print out update (small amount of device info is transferred).
#define RUNSERIAL 0 //driver.c
#define SERIAL_ITER 5 //driver.c

#define TPB2 8 //Only for kernel 4+. Thread dimension for square block (threads = TPB2*TPB2; should be multiple of 32!)

#define TSD 5 //Only for kernel 4+. Thread swap dimension; Dimension of block of swaps each thread performs.
		//The number of coordinates transferred are 2*(TSD+1). The number of dtype required is 2*(2*(TSD+1)).
		//So 4*(TSD+1) 32-bit registers are required for dtype=float.

#define BLOCKLIMIT 65000 //Max number of blocks per kernel. Dictated by hardware version.

//reduction and rearrangment parameters (after main kernel).
#define REDUCE_TPB 256
#define SWAP_TPB 256


//Output controls
#define OUTPUT_BEST_IMPROVEMENT 1 //Per iteration, the best improvement result will fprintf to a file.


//Derived variables
#define NCOORDS ((TSD)+1) //number of coordinates to copy, for use with TSD (kernels 4+)
#define BSD (TPB2*TSD)//Block swap dimension, for use with TSD (kernels 4+)



//old params
#define TPB 256 // Threads per block. Redefined here, because setting to 256 or lower breaks the other versions due to too many blocks made. 
#define TPB2_3 16 //TPB2 for kernel 3X. Thread dimension for square block (threads = TPB2*TPB2; should be multiple of 32!)
#define ADDS 6 // make sure the while loops available are compatible with this number.
#define SEQUENTIAL 1 //0 or 1, determines where the cpu reduction takes place. 
					//0: cpu reduction is not in GPU timing. 1: cpu reduction is in GPU timing.
#define RUNALL 1 //0 or 1; 0: run a single kernel as specified by VERSION. 1: run all kernels.

