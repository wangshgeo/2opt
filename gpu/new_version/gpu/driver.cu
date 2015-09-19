#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "Instance.hh"
#include "StopWatch.hh"

#include "cuda_utils.hh"

#include "parameters.hh"

#include "cub-1.4.1/cub/cub.cuh"
#include "cub-1.4.1/cub/block/block_load.cuh"
#include "cub-1.4.1/cub/block/block_store.cuh"
#include "cub-1.4.1/cub/block/block_reduce.cuh"

#include "kernel.cuh"

__global__ void 
gpuSwapReduce (dtype *d_differences,dtype *d_differences2,int *d_mink,int *d_mink2,int *d_ij,int nd);

__global__ void 
gpuSwap2(dtype* d_x, dtype* d_y, int* d_ij);

__global__ void
gpuSwap3 (dtype *d_x, dtype* d_y, cub::KeyValuePair<int, dtype> *d_out, 
	dtype* d_differences, int* d_k,int*d_ij);

__global__ void
gpuSwap4 (dtype *d_x, dtype* d_y, 
	cub::KeyValuePair<int, dtype> *d_out, 
	dtype* d_differences, int* d_k,
	int* d_i,int *d_j,
	cost_t *d_segment_lengths);

__global__ void
updateLengths (dtype *d_x, dtype* d_y, 
	cub::KeyValuePair<int, dtype> *d_out, 
	int* d_i,int *d_j,
	cost_t *d_segment_lengths);

void getCudaInts(int*d_A, int*h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (
		cudaMemcpy (h_A,d_A,N*sizeof(int), cudaMemcpyDeviceToHost)
		);
}

void getCudaDtype(dtype*d_A, dtype*h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (
		cudaMemcpy (h_A,d_A,N*sizeof(dtype), cudaMemcpyDeviceToHost)
		);
}



int main(int argc, char ** argv)
{
	if(argc < 2)
	{
		fprintf(stderr,"Please enter an input file!\n");
		return EXIT_FAILURE;
	}

	int random_seed = 0;
	srand(random_seed);

	char* city_file_name = argv[1];
	fprintf(stdout,"Reading file at: %s\n",city_file_name);
	Instance instance(city_file_name);
	int nc = instance.getCityCount();
	fprintf(stdout,"City count: %d\n\n",nc);
	fprintf(stdout,"Done reading file.\n");

	fprintf(stdout,"Initializing (Host) Initial Tours... ");
	int *h_initialTour;
	h_initialTour = new int[nc];
	for(int i=0;i<nc;++i) h_initialTour[i] = i;
	random_shuffle(h_initialTour,h_initialTour+nc);
	fprintf(stdout,"Done.\n");
	
	fprintf(stdout,"Initializing (Host) Coordinates... ");
	dtype *h_x,*h_y;
	h_x = instance.getX();
	h_y = instance.getY();
	dtype* ordered_x = new dtype[nc];
	dtype* ordered_y = new dtype[nc];
	for(int i=0;i<nc;++i)
	{
		ordered_x[i]=h_x[h_initialTour[i]];
		ordered_y[i]=h_y[h_initialTour[i]];
	}
	for(int i=0;i<nc;++i)
	{
		h_x[i]=ordered_x[i];
		h_y[i]=ordered_y[i];
	}
	delete[] ordered_x;
	delete[] ordered_y;
	fprintf(stdout,"Done.\n");

	fprintf(stdout,"Initializing GPU data...");
	int *d_initialTour = NULL;
	dtype *d_x = NULL;
	dtype *d_y = NULL;
	dtype* x2 = new dtype[nc+THREAD_CHUNK*BLOCK_DIMENSION];
	dtype* y2 = new dtype[nc+THREAD_CHUNK*BLOCK_DIMENSION];
	for(int i=0;i<nc;++i)
	{
		x2[i] = h_x[i];
		y2[i] = h_y[i];
	}
	for(int i=nc;i<nc+THREAD_CHUNK*BLOCK_DIMENSION;++i)
	{
		x2[i] = 0;
		y2[i] = 0;
	}
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_initialTour, nc*sizeof(int)));
	CUDA_CHECK_ERROR (cudaMemcpy (d_initialTour, h_initialTour, nc*sizeof(int),cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_x, (nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (d_x, x2, (nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(dtype),cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_y, (nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (d_y, y2, (nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(dtype),cudaMemcpyHostToDevice));
	cost_t *h_segment_lengths = new cost_t[nc+THREAD_CHUNK*BLOCK_DIMENSION];
	for(int i=0;i<nc-1;++i)
	{
		dtype dx = h_x[i+1] - h_x[i];
		dtype dy = h_y[i+1] - h_y[i];
		h_segment_lengths[i] = sqrt(dx*dx + dy*dy);
	}
	dtype dx = h_x[0] - h_x[nc-1];
	dtype dy = h_y[0] - h_y[nc-1];
	h_segment_lengths[nc-1] = sqrt(dx*dx + dy*dy);
	for(int i=nc;i<nc+THREAD_CHUNK*BLOCK_DIMENSION;++i)
	{
		h_segment_lengths[i] = 0;
	}
	cost_t *d_segment_lengths = NULL;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_segment_lengths, (nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(cost_t)));
	CUDA_CHECK_ERROR (cudaMemcpy (d_segment_lengths, h_segment_lengths, 
		(nc+THREAD_CHUNK*BLOCK_DIMENSION)*sizeof(cost_t),cudaMemcpyHostToDevice));
	fprintf(stdout," Done.\n");

	fprintf(stdout,"Determining Thread and Block Dimensions...");
	int N = ((nc-3)*(nc-2))/2;//Number of swaps to check (each thread does >= 1)
	int ND = -1;//Number of differences output from GPU to min-reduce. 
	int GX=1,GY=1;//block dimensions of grid.
	int BX=1,BY=1;//thread dimensions of block.
	int NN = nc-3;//such that n_swaps = NN*(NN+1)/2
	int SBB = TPB2*TSD;//Block dimension terms of swaps. SBB = BB if swaps per thread is 1.
	int BB2 =(NN+SBB-1)/SBB;///Block dimension of the grid, for TPB2-sized square blocks WITH TSD^2 swaps performed per thread (versus regular 1).
	int gridSplits = -1;//for when the problem gets too large for 65535 blocks.
	int lastGX = -1;//if gridSplits > 0, this is the number of blocks in the last split (less than BLOCKLIMIT)
	int totalBlocks = -1;


	GX = BB2*(BB2+1)/2;
	BX = TPB2;
	BY = TPB2;
	ND = GX;
	totalBlocks = BB2*(BB2+1)/2;
	//int swapChecksPerThread = TSD*TSD;
	
	//int swapChecksPerBlock = BX*BY*swapChecksPerThread;
	gridSplits = totalBlocks/BLOCKLIMIT;

	GX=BLOCKLIMIT;
	dim3 grid (GX,GY);
	dim3 block (BX,BY);
	lastGX = totalBlocks-gridSplits*BLOCKLIMIT;
	dim3 lastGrid (lastGX,GY);

	//Preprocessing for the gpu second reduction
	int cityblocks = (nc+SWAP_TPB-1)/SWAP_TPB;
	dim3 grid2 (cityblocks,1);
	dim3 block2 (SWAP_TPB,1);
	//end

	//Output information about run time parameters.
	fprintf (stdout, "Total Number of Blocks: %d\n",totalBlocks);
	fprintf (stdout, "Limited Grid Dim: %d x %d\n",GX,GY);
	fprintf (stdout, "Last Grid Dim: %d x %d\n",lastGX,GY);
	fprintf (stdout, "Block Dim: %d x %d\n",BX,BY);
	fprintf (stdout, "Size of Returned Array: %d\n",ND);
	fprintf (stdout, "Number of swaps computed and reduced: %d\n",N);
	//End output

	//Resource allocation for reduction steps.
	dtype *h_differences = new dtype[ND]; 
	dtype *d_differences,*d_differences2;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences, ND * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences2, ND * sizeof (dtype)));
	cub::KeyValuePair<int, dtype> *d_out;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_out, 1 * sizeof (cub::KeyValuePair<int, dtype>)));
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_differences, d_out, ND);
	CUDA_CHECK_ERROR (cudaMalloc(&d_temp_storage, temp_storage_bytes));
	// int *h_mink = new int[ND]; 
	int *d_mink,*d_mink2;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink, ND * sizeof (int)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink2, ND * sizeof (int)));
	int *d_ij;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_ij, 2 * sizeof (int)));
	//New kernel stuff.
	int block_swap_dimension = BLOCK_DIMENSION * THREAD_CHUNK;
	int base_blocks = ( nc + block_swap_dimension - 1 ) / block_swap_dimension;
	int total_blocks = base_blocks * (base_blocks + 1) / 2;
	cost_t *d_cost = NULL;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_cost, total_blocks * sizeof (cost_t)));
	int *d_i = NULL;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_i, total_blocks * sizeof (int)));
	int *d_j = NULL;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_j, total_blocks * sizeof (int)));

	//end new kernel stuff.

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);//Shared Memory <-> L1 Cache setting.

	//Timer initialization and start
	cudaEvent_t start, stop;
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	//end

	//EXECUTE KERNEL
	int iterations;
	int k_best;
	dtype diff_best;
	for(iterations = 0; iterations < MAX_ITER; ++iterations)
	{
		//Single best-improvement iteration
		cudaDeviceSynchronize();
		for(int j=0;j<gridSplits;++j)
		{
			kernel <<<grid, block>>> (d_x, d_y, d_differences, d_mink,
				nc, BB2, j*BLOCKLIMIT);
		}
		kernel  <<<lastGrid, block>>> (d_x, d_y, d_differences, d_mink,
			nc, BB2, gridSplits*BLOCKLIMIT);
		//end single iteration

		//Single best-improvement iteration
		// cudaDeviceSynchronize();
		// int d_index = 0;
		// for(int i=0;i<base_blocks;++i)
		// {
		// 	dim3 grid (base_blocks-i,1);
		// 	dim3 block (BLOCK_DIMENSION,BLOCK_DIMENSION);
		// 	kernel2 <<<grid, block>>>(d_x, d_y, d_segment_lengths,
		// 		nc, i*THREAD_CHUNK*BLOCK_DIMENSION, i*THREAD_CHUNK*BLOCK_DIMENSION,
		// 		d_cost, d_i, d_j, d_index);
		// 	d_index += base_blocks-i;
		// }
		//end single iteration
		
		//gpu reduce
		cudaDeviceSynchronize();
		cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_differences, d_out, ND);
		// cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_cost, d_out, ND);

		//gpu tour rearrangement
		cudaDeviceSynchronize();
		gpuSwap3<<<grid2,block2>>>(d_x,d_y,d_out,d_differences,d_mink,d_ij);
		// gpuSwap4<<<grid2,block2>>>(d_x, d_y, d_out, d_differences, d_mink, d_i, d_j, d_segment_lengths);
		// cudaDeviceSynchronize();
		// updateLengths<<<1,2>>>(d_x, d_y, d_out, d_i, d_j, d_segment_lengths);
		// cudaDeviceSynchronize();
		//end

		//post-processing and optional cpu tour rearrangement
		if(iterations%REFRESH_ITER==0)
		{
			cudaDeviceSynchronize();
			getCudaDtype(d_differences,&diff_best,1);
			getCudaInts(d_mink,&k_best,1);
			int ij[2];
			getCudaInts(d_ij,ij,2);
			cudaDeviceSynchronize();
			//int ij[2];
			//ijfromk(bi.k,ij);
			if(diff_best < 0)
			{
				//cpu tour rearrangement
				//swapDeviceTour(h_coords,d_coords,nc,ij);
				fprintf (stdout, "Iteration %d GPU min diff,i,j,raw k : %f,%d,%d,%d\n",iterations,diff_best,ij[0],ij[1],k_best);
			}
			else
			{
				fprintf (stdout, "No more improvements found! Stopping iteration.\n"); 
				break;
			}
		}
		//end
	}
	//END KERNEL EXECUTION

	//Timer stop, get time, destroy timer resources
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	float elapsedTimePerIteration;
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTimePerIteration, start, stop));
	elapsedTimePerIteration /= iterations;
	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

	// //Check tour correctness.
	// checkCoordValidity(h_coords,original,nc);
	// //end

	//Output the timing and performance results.
	fprintf (stdout, "GPU min diff, raw k : %f, %d\n",diff_best,k_best);
	fprintf (stdout, "Execution time per iteration: %f ms\n", elapsedTimePerIteration);
	fprintf (stdout, "Swap checks per iteration: %d\n", N);
	fprintf (stdout, "Speed (Gmoves/s): %f\n\n",N/(elapsedTimePerIteration/1000.0)/1e9);
	fprintf (stdout, "Threads Per Block: %d x %d\n",BX,BY);
	//fprintf (stdout, "Equivalent performance: %f GB/s\n", (N * sizeof (dtype) / elapsedTime) * 1e-6);



	cudaFree(d_initialTour);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_segment_lengths);

	return EXIT_SUCCESS;
}
