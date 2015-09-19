#include "Instance.hh"
#include "cuda_utils.hh"

#include "cost.hh"
#include "parameters.hh"

#include "cub-1.4.1/cub/cub.cuh"
#include "cub-1.4.1/cub/block/block_load.cuh"
#include "cub-1.4.1/cub/block/block_store.cuh"
#include "cub-1.4.1/cub/block/block_reduce.cuh"
#include "cub-1.4.1/cub/block/specializations/block_reduce_raking_commutative_only.cuh"

__device__ int kfromij(int i,int j)
{
	i-=1;
	j-=2;
	return i+((j*(j+1))>>1);
}

__global__ void 
kernel(const dtype *d_x, const dtype *d_y, dtype *d_diff,int *d_k, 
	const unsigned int nc, const unsigned int BB, const int globalStart)
{//Version 9. Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.
	dtype ii_x[NCOORDS];
	dtype ii_y[NCOORDS];
	dtype jj_x[NCOORDS];
	dtype jj_y[NCOORDS];

	dtype savedi[TSD];//saved distances corresponding to i (i-1 to i)
	dtype savedj[TSD];//saved distances corresponding to j (j to j+1)
	dtype savedij[TSD+1][TSD+1];//saved distances (i-1 -> j)

	//Calculate the indices needed for all later calculations.
	unsigned int maxn = nc-3;
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	unsigned int istart = bi*BSD+1;//i start from block
	istart+=TSD*threadIdx.x;//i start from thread
	int jstart =  (nc-2)-(bj+1)*BSD+1;//j start from block
	jstart+=TSD*threadIdx.y;//j start from thread

	//Now transfer the memory from global into registers.
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int id=istart-1+c;
		if(id<maxn+1)
		{
			ii_x[c] = d_x[id];
			ii_y[c] = d_y[id];
		}
	}
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj_x[c] = d_x[id];
			jj_y[c] = d_y[id];
		}
	}
	// #pragma unroll
	for(int i=0;i<TSD;++i)
	{
		savedi[i]=distanceEUC_2D(ii_x[i],ii_y[i],ii_x[i+1],ii_y[i+1]);
	}
	// #pragma unroll
	for(int j=0;j<TSD;++j)
	{
		savedj[j]=distanceEUC_2D(jj_x[j],jj_y[j],jj_x[j+1],jj_y[j+1]);
	}
	// #pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		// #pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D(ii_x[sx],ii_y[sx],jj_x[sy],jj_y[sy]);
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	int mink = -1;
	// #pragma unroll
	for(int sx=0;sx<TSD;++sx)
	{
		// #pragma unroll
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype dold = savedi[sx]+savedj[sy];
				// if (savedij[sx][sy] > dold) continue;
				// if (savedij[sx+1][sx+1] > dold) continue;
				dtype dnew = savedij[sx][sy]+savedij[sx+1][sy+1];
				//dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
				//		distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				dtype difference = dnew-dold;
				mink = (difference<min)?(kfromij(gi,gj)):mink;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

    typedef cub::BlockReduce<dtype, TPB2, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, TPB2, 1, 200> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    dtype aggregate = BlockReduce(temp_storage).Reduce(min,cub::Min());
	__shared__ dtype min_block;
	int tid = threadIdx.x+threadIdx.y*TPB2;
	if (tid == 0){
		min_block = aggregate;
	}
	__syncthreads();
	if (min_block == min)
	{
		d_k[block] = mink;
		d_diff[block] = min;
	}
}


// __global__ void 
// kernel2(const dtype *d_x, const dtype *d_y, const cost_t *d_segment_lengths,
// 	const unsigned int nc, const int i_block_start, const int j_block_start,
// 	cost_t *d_cost,int *d_i,int *d_j,const int d_index_start)
// {//Revamped version. Less confusing indices.
// 	//Also will try utilizing global memory more, since we are compute-bound.
// 	//Remember to buffer the device arrays!!!
// 	dtype i_x[THREAD_CHUNK+1];
// 	dtype i_y[THREAD_CHUNK+1];
// 	dtype j_x[THREAD_CHUNK+1];
// 	dtype j_y[THREAD_CHUNK+1];

// 	cost_t i_segments[THREAD_CHUNK];
// 	cost_t j_segments[THREAD_CHUNK];
// 	// dtype ij_segments[THREAD_CHUNK][THREAD_CHUNK];//saved distances (i-1 -> j)

// 	int i_start = i_block_start + threadIdx.y*THREAD_CHUNK;
// 	int j_start = j_block_start + 
// 		blockIdx.x*THREAD_CHUNK*BLOCK_DIMENSION + 
// 		threadIdx.x*THREAD_CHUNK;

// 	#pragma unroll
// 	for(int k=0;k<THREAD_CHUNK;++k)
// 	{
// 		i_x[k] = d_x[i_start+k];
// 		i_y[k] = d_y[i_start+k];
// 		j_x[k] = d_x[j_start+k];
// 		j_y[k] = d_y[j_start+k];
// 		i_segments[k] = d_segment_lengths[i_start+k];
// 		j_segments[k] = d_segment_lengths[j_start+k];
// 	}
// 	i_x[THREAD_CHUNK] = d_x[i_start+THREAD_CHUNK];
// 	i_y[THREAD_CHUNK] = d_y[i_start+THREAD_CHUNK];
// 	j_x[THREAD_CHUNK] = d_x[j_start+THREAD_CHUNK];
// 	j_y[THREAD_CHUNK] = d_y[j_start+THREAD_CHUNK];


// 	cost_t best_cost = 0;
// 	int i_best = -1;
// 	int j_best = -1;
// 	#pragma unroll
// 	for(int i=0;i<THREAD_CHUNK;++i)
// 	{
// 		#pragma unroll
// 		for(int j=0;j<THREAD_CHUNK;++j)
// 		{
// 			int i_global = i_start+i;
// 			int j_global = j_start+j;
// 			bool active = i_global <= j_global - 2;
			
// 			cost_t current_cost = (active) ? (i_segments[i] + j_segments[j]) : 0;
// 			dtype dx = (active) ? (j_x[j]-i_x[i]) : 0;
// 			dtype dy = (active) ? (j_y[j]-i_y[i]) : 0;
// 			cost_t new_cost = (active) ? round(sqrt(dx*dx + dy*dy)) : 0;
// 			bool candidate = ((new_cost < current_cost) and active) ? true : false;
// 			dx = (not candidate) ? 0 : j_x[j+1]-i_x[i+1];
// 			dy = (not candidate) ? 0 : j_y[j+1]-i_y[i+1];
// 			new_cost = (not candidate) ? 0 : new_cost + round(sqrt(dx*dx + dy*dy));
// 			bool better = ((new_cost - current_cost) < best_cost) and candidate;
// 			best_cost = (better) ? (new_cost - current_cost) : best_cost;
// 			i_best = (better) ? i_global : i_best;
// 			j_best = (better) ? j_global : j_best;
// 		}
// 	}

//     typedef cub::BlockReduce<cost_t, BLOCK_DIMENSION, 
//     	cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BLOCK_DIMENSION, 1, 200> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     cost_t aggregate = BlockReduce(temp_storage).Reduce(best_cost,cub::Min());
// 	__shared__ cost_t best_cost_shared;
// 	if (threadIdx.x == 0 and threadIdx.y == 0){
// 		best_cost_shared = aggregate;
// 	}
// 	__syncthreads();
// 	if ( best_cost == best_cost_shared )
// 	{
// 		d_i[d_index_start + blockIdx.x] = i_best;
// 		d_j[d_index_start + blockIdx.x] = j_best;
// 		d_cost[d_index_start + blockIdx.x] = best_cost;
// 	}
// }

/*
__global__ void 
kernel9 (COORD *d_coords, dtype *d_differences,int *d_mink, 
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 9. Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];
	dtype savedi[TSD];//saved distances corresponding to i (i-1 to i)
	dtype savedj[TSD];//saved distances corresponding to j (j to j+1)
	dtype savedij[TSD+1][TSD+1];//saved distances (i-1 -> j)

	//Calculate the indices needed for all later calculations.
	unsigned int maxn = nc-3;
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	unsigned int istart = bi*BSD+1;//i start from block
	istart+=TSD*threadIdx.x;//i start from thread
	int jstart =  (nc-2)-(bj+1)*BSD+1;//j start from block
	jstart+=TSD*threadIdx.y;//j start from thread

	//Now transfer the memory from global into registers.
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int id=istart-1+c;
		if(id<maxn+1)
		{
			ii[c] = d_coords[id];
		}
	}
	#pragma unroll
	for(int i=0;i<TSD;++i)
	{
		savedi[i]=distanceEUC_2D(ii[i].x,ii[i].y,ii[i+1].x,ii[i+1].y);
	}
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj[c] = d_coords[id];
		}
	}
	#pragma unroll
	for(int j=0;j<TSD;++j)
	{
		savedj[j]=distanceEUC_2D(jj[j].x,jj[j].y,jj[j+1].x,jj[j+1].y);
	}
	#pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y);
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	int mink = -1;
	#pragma unroll
	for(int sx=0;sx<TSD;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype dold = savedi[sx]+savedj[sy];
				dtype dnew = savedij[sx][sy]+savedij[sx+1][sy+1];
				//dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
				//		distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				dtype difference = dnew-dold;
				mink = (difference<min)?(kfromij(gi,gj)):mink;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2*TPB2];
	__shared__ int smink[TPB2*TPB2];
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	smink[tid] = mink;
	__syncthreads();
	int check=-1;
	dtype currentsdata = sdata[tid];
	dtype nextsdata;
	int currentsmink = smink[tid];
	int nextsmink;
	if (TPB2*TPB2 >= 1024){ if (tid < 512) {
		//check = sdata[tid+512]<sdata[tid];
		nextsdata = sdata[tid+512];
		nextsmink = smink[tid+512];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) {
		nextsdata = sdata[tid+256];
		nextsmink = smink[tid+256];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) {
		nextsdata = sdata[tid+128];
		nextsmink = smink[tid+128];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) {
		nextsdata = sdata[tid+64];
		nextsmink = smink[tid+64];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (tid < 32) warpMinReduce2(sdata,smink,tid);
	if (tid == 0){
		d_differences[block] = sdata[0];
		d_mink[block] = smink[0];
	}
}

__global__ void 
kernel10 (COORD *d_coords, dtype *d_differences,int *d_mink, 
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 10. Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.
	//CURRENTLY BROKEN!
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];
	dtype savedi[TSD];//saved distances corresponding to i (i-1 to i)
	dtype savedj[TSD];//saved distances corresponding to j (j to j+1)
	dtype savedij[TSD+1][TSD+1];//saved distances (i-1 -> j)

	//Calculate the indices needed for all later calculations.
	int maxn = nc-3;
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	int istart0 = bi*BSD+1;//i start from block
	int istart = istart0 + TSD*threadIdx.x;//i start from thread
	int jstart0 =  (nc-2)-(bj+1)*BSD+1;//j start from block
	int jstart = jstart0 + TSD*threadIdx.y;//j start from thread

	//Now transfer the memory from global into registers.
	__shared__ COORD sharedi[TSD*TPB2*TPB2];
	#pragma unroll
	for(int t=0;t<TSD;++t)
	{
		int li = t*blockDim.x + threadIdx.x;
		int id = istart0 - 1 + li;
		if(id < maxn+1)
		{
			sharedi[li] = d_coords[id];
		}
	}
	#pragma unroll
	for(int i=0;i<NCOORDS;++i)
	{
		int id = TSD*threadIdx.x + i;
		if(id < TSD*blockDim.x)
			ii[i] = sharedi[id];
	}
	// //old
	// #pragma unroll
	// for(int c=0;c<NCOORDS;++c)
	// {
	// 	int id=istart-1+c;
	// 	if(id<maxn+1)
	// 	{
	// 		ii[c] = d_coords[id];
	// 	}
	// }
	// //end
	#pragma unroll
	for(int i=0;i<TSD;++i)
	{
		savedi[i]=distanceEUC_2D(ii[i].x,ii[i].y,ii[i+1].x,ii[i+1].y);
	}
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj[c] = d_coords[id];
		}
	}
	#pragma unroll
	for(int j=0;j<TSD;++j)
	{
		savedj[j]=distanceEUC_2D(jj[j].x,jj[j].y,jj[j+1].x,jj[j+1].y);
	}
	#pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y);
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	int mink = -1;
	#pragma unroll
	for(int sx=0;sx<TSD;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype dold = savedi[sx]+savedj[sy];
				dtype dnew = savedij[sx][sy]+savedij[sx+1][sy+1];
				//dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
				//		distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				dtype difference = dnew-dold;
				mink = (difference<min)?(kfromij(gi,gj)):mink;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2*TPB2];
	__shared__ int smink[TPB2*TPB2];
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	smink[tid] = mink;
	__syncthreads();
	int check=-1;
	dtype currentsdata = sdata[tid];
	dtype nextsdata;
	int currentsmink = smink[tid];
	int nextsmink;
	if (TPB2*TPB2 >= 1024){ if (tid < 512) {
		//check = sdata[tid+512]<sdata[tid];
		nextsdata = sdata[tid+512];
		nextsmink = smink[tid+512];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) {
		nextsdata = sdata[tid+256];
		nextsmink = smink[tid+256];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) {
		nextsdata = sdata[tid+128];
		nextsmink = smink[tid+128];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) {
		nextsdata = sdata[tid+64];
		nextsmink = smink[tid+64];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (tid < 32) warpMinReduce2(sdata,smink,tid);
	if (tid == 0){
		d_differences[block] = sdata[0];
		d_mink[block] = smink[0];
	}
}

__global__ void 
kernel11 (COORD *d_coords, dtype *d_differences,int *d_mink, 
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 11; Version 9 but with corrections to alleviate the NVPROF-diagnosed global memory problems. 
	//  Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];
	dtype savedi[TSD];//saved distances corresponding to i (i-1 to i)
	dtype savedj[TSD];//saved distances corresponding to j (j to j+1)
	dtype savedij[TSD+1][TSD+1];//saved distances (i-1 -> j)

	//Calculate the indices needed for all later calculations.
	unsigned int maxn = nc-3;
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	unsigned int istart = bi*BSD+1;//i start from block
	istart+=TSD*threadIdx.x;//i start from thread
	int jstart =  (nc-2)-(bj+1)*BSD+1;//j start from block
	jstart+=TSD*threadIdx.y;//j start from thread

	//Now transfer the memory from global into registers.
	#pragma unroll
	for(int c=0;c<NCOORDS;++c)
	{
		int idi=istart-1+c;
		if(idi<maxn+1)
		{
			ii[c] = d_coords[idi];
		}
		int idj=jstart+c;
		if(idj<nc-bj*BSD && idj>0)
		{
			jj[c] = d_coords[idj];
		}
	}
	#pragma unroll
	for(int i=0;i<TSD;++i)
	{
		savedi[i]=distanceEUC_2D(ii[i].x,ii[i].y,ii[i+1].x,ii[i+1].y);
	}
	#pragma unroll
	for(int j=0;j<TSD;++j)
	{
		savedj[j]=distanceEUC_2D(jj[j].x,jj[j].y,jj[j+1].x,jj[j+1].y);
	}
	#pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y);
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	int mink = -1;
	#pragma unroll
	for(int sx=0;sx<TSD;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype dold = savedi[sx]+savedj[sy];
				dtype dnew = savedij[sx][sy]+savedij[sx+1][sy+1];
				//dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
				//		distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				dtype difference = dnew-dold;
				mink = (difference<min)?(kfromij(gi,gj)):mink;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2*TPB2];
	__shared__ int smink[TPB2*TPB2];
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	smink[tid] = mink;
	__syncthreads();
	int check=-1;
	dtype currentsdata = sdata[tid];
	dtype nextsdata;
	int currentsmink = smink[tid];
	int nextsmink;
	if (TPB2*TPB2 >= 1024){ if (tid < 512) {
		//check = sdata[tid+512]<sdata[tid];
		nextsdata = sdata[tid+512];
		nextsmink = smink[tid+512];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) {
		nextsdata = sdata[tid+256];
		nextsmink = smink[tid+256];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) {
		nextsdata = sdata[tid+128];
		nextsmink = smink[tid+128];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) {
		nextsdata = sdata[tid+64];
		nextsmink = smink[tid+64];
		check = nextsdata<currentsdata;
		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
		__syncthreads(); }
	if (tid < 32) warpMinReduce2(sdata,smink,tid);
	if (tid == 0){
		d_differences[block] = sdata[0];
		d_mink[block] = smink[0];
	}
}











void gpuSwapCheck(COORD* h_tour,int nc)
{//swaps h_tour.
	COORD* d_tour;
	initCudaCoords(&d_tour,h_tour,nc);
	
	int ij[2] = { nc/5, 5*nc/6 };
	int* d_ij;
	initCudaTour(&d_ij,ij,2);
	
	int tpb = 256;
	int blocks;
	//require host knowledge
	int diff = ij[1]-ij[0];
	blocks = (diff+tpb-1)/tpb;
	dim3 grid(blocks,1);
	dim3 block(tpb,1);
	//gpuSwap<<<grid,block>>>(d_tour,d_ij);
	//end
	//dont require host knowledge
	blocks = (nc+tpb-1)/tpb;
	dim3 grid2(blocks,1);
	dim3 block2(tpb,1);
	gpuSwap2<<<grid2,block2>>>(d_tour,d_ij);
	cudaDeviceSynchronize();

	swapCoords(h_tour,ij);

	COORD tourFromGpu[nc];
	getCudaCoords(d_tour,tourFromGpu,nc);

	fprintf(stderr,"Comparing the gpu and cpu swap implementations...\n");
	for(int i=0;i<nc;++i)
	{
		if((tourFromGpu[i].x != h_tour[i].x) or (tourFromGpu[i].y != h_tour[i].y))
		{
			fprintf(stderr,"Mismatch detected at index %d!\n",i);
		}
	}
	fprintf(stderr,"Done!\n");

	cudaFree(d_tour);
	cudaFree(d_ij);
}

void checkCoordValidity(COORD*newTour,COORD*originalTour,int nc)
{
	fprintf(stderr,"Checking the validity of the new tour...\n");
	int occurences[nc];
	int repetitions[nc];
	//initialize occurences.
	for(int i=0;i<nc;++i)
	{
		occurences[i] = 0;
		repetitions[i] = -1;
	}
	//repetition within new tour.
	for(int i=0;i<nc;++i)
	{
		dtype x = newTour[i].x;
		dtype y = newTour[i].y;
		for(int j=0;j<nc;++j)
		{
			if((x == newTour[j].x) and (y == newTour[j].y))
			{
				++repetitions[i];
			}
		}
		if(repetitions[i] > 0)
		{
			fprintf(stderr,"Error! City %d is repeated in new tour!\n",i);
		}
	}
	//check that every city in original tour is accounted for.
	for(int i=0;i<nc;++i)
	{
		dtype x = newTour[i].x;
		dtype y = newTour[i].y;
		//go through the original tour.
		for(int j=0;j<nc;++j)
		{
			if((x == originalTour[j].x) and (y == originalTour[j].y))
			{
				++occurences[i];
			}
		}
		if(occurences[i] > 1)
		{
			fprintf(stderr,"Error! City %d was used more than once!\n",i);
		}
		if(occurences[i] < 1)
		{
			fprintf(stderr,"Error! City %d was not used!\n",i);
		}
	}
	fprintf(stderr,"Tour validity check done!\n");
}

typedef struct bestimprovement
{
	int k;
	dtype diff;
} BESTIMPROVEMENT;

BESTIMPROVEMENT minReduceCpu2(dtype*h,dtype*d,int*h_mink,int*d_mink,int n)
{
	CUDA_CHECK_ERROR (cudaMemcpy (h, d, n * sizeof (dtype), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR (cudaMemcpy (h_mink, d_mink, n * sizeof (int), cudaMemcpyDeviceToHost));
	//A minimum reduction.
	dtype minDifference=0;
	int minI = -1;
	for(int i=0; i<n; i++)
	{
		if(h[i] < minDifference)
		{
			minDifference = h[i];
			minI = h_mink[i];
		}
	}
	BESTIMPROVEMENT ret = {minI,minDifference};
	return ret;
}

void swapDeviceTour(COORD*h_tour,COORD*d_tour,int nc,int swapij[])
{//swaps host tour, then uploads to device.
	swapCoords(h_tour,swapij);
	transferCudaCoords(d_tour,h_tour,nc);
}
dtype coordDistance(COORD c1,COORD c2)
{
	dtype dx = c2.x-c1.x;
	dtype dy = c2.y-c1.y;
	return sqrt(dx*dx+dy*dy);
}
dtype tourLength(COORD*h_tour,int nc)
{
	dtype total=0;
	for(int i=0;i<nc;++i)
	{
		total+=coordDistance(h_tour[i],h_tour[(i+1)%nc]);
	}
	return total;
}
void cudaOpt2 (COORD *h_coords,COORD *d_coords, unsigned int nc)
{//d_coords and h_coords evolve.
	//Set and determine thread and block dimensions
	int N = ((nc-3)*(nc-2))/2;//Number of swaps to check (each thread does >= 1)
	//int NT = N;//Number of Threads.
	int ND = -1;//Number of differences output from GPU to min-reduce. 
	int GX=1,GY=1;//block dimensions of grid.
	int BX=1,BY=1;//thread dimensions of block.
	int NN = nc-3;//such that N = NN*(NN+1)/2
	int SBB = TPB2*TSD;//Block dimension  terms of swaps. SBB = BB if swaps per thread is 1.
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
	fprintf (stderr, "Total Number of Blocks: %d\n",totalBlocks);
	fprintf (stderr, "Limited Grid Dim: %d x %d\n",GX,GY);
	fprintf (stderr, "Last Grid Dim: %d x %d\n",lastGX,GY);
	fprintf (stderr, "Block Dim: %d x %d\n",BX,BY);
	fprintf (stderr, "Size of Returned Array: %d\n",ND);
	fprintf (stderr, "Number of swaps computed and reduced: %d\n",N);
	//End output

	//Resource allocation.
	//dtype h_differences[ND];
	dtype *h_differences = new dtype[ND]; 
	dtype *d_differences,*d_differences2;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences, ND * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences2, ND * sizeof (dtype)));
	int *h_mink = new int[ND]; 
	int *d_mink,*d_mink2;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink, ND * sizeof (int)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink2, ND * sizeof (int)));
	int *d_ij;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_ij, 2 * sizeof (int)));
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);//Shared Memory <-> L1 Cache setting.

	//Copy for correctness check.
	COORD original[nc];
	for(int i=0;i<nc;++i) { original[i] = h_coords[i]; }
	//Copy end.

	//gpu swap check
	//COORD h_swapcheck[nc];
	//for(int i=0;i<nc;++i) { h_swapcheck[i] = h_coords[i]; }
	//gpuSwapCheck(h_swapcheck,nc);
	//end check

	//Timer initialization and start
	cudaEvent_t start, stop;
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	//end

	//EXECUTE KERNEL
	BESTIMPROVEMENT bi = {0,0};
	//Run the kernel for some iterations
	//dtype ptl = tourLength(h_coords,nc);
	int iterations;
	for(iterations = 0; iterations < MAX_ITER; ++iterations) {
		//Single best-improvement iteration
		cudaDeviceSynchronize();
		for(int j=0;j<gridSplits;++j)
		{
			kernel11 <<<grid, block>>> (d_coords, d_differences,d_mink,
				nc,BB2,j*BLOCKLIMIT);
		}
		kernel11 <<<lastGrid, block>>> (d_coords, d_differences,d_mink,
			nc,BB2,gridSplits*BLOCKLIMIT);
		//end single iteration
		
		//Compute and display tour length.
		//dtype tl = tourLength(h_coords,nc);
		//fprintf (stderr, "Current tour length %f (difference from previous: %f)\n",tl,tl-ptl);
		//ptl = tl;
		//end

/*
		//cpu reduce
		cudaDeviceSynchronize();
		bi = minReduceCpu2(h_differences,d_differences,h_mink,d_mink,ND);
		int ij[2];
		ijfromk(bi.k,ij);
		putCudaInts(d_ij,ij,2);
		//end


		//gpu reduce
		int elements = ND;
		int mode = 1;
		while(elements > 1)
		{
			//fprintf(stderr,"elements: %d\n",elements);
			int rblocks = (elements+REDUCE_TPB-1)/REDUCE_TPB;
			dim3 rgrid (rblocks,1);
			dim3 rblock (REDUCE_TPB,1);
			cudaDeviceSynchronize();
			//fprintf(stderr,"rblocks: %d\n",rblocks);
			if(mode==1)
			{
				gpuSwapReduce<<<rgrid,rblock>>>(d_differences,d_differences2,d_mink,d_mink2,d_ij,elements);
				mode=0;
			}
			else
			{
				gpuSwapReduce<<<rgrid,rblock>>>(d_differences2,d_differences,d_mink2,d_mink,d_ij,elements);
				mode=1;
			}
			elements=rblocks;
		}
		// if(mode==0)
		// {
		// 	dim3 grid3 (1,1);
		// 	dim3 block3 (1,1);
		// 	cudaDeviceSynchronize();
		// 	correctReduceBuffer<<<grid3,block3>>>(d_differences,d_differences2,d_mink,d_mink2);
		// }
		//end

		//gpu tour rearrangement
		cudaDeviceSynchronize();
		//fprintf(stderr,"grid2, block2: %d, %d\n",cityblocks,SWAP_TPB);
		gpuSwap2<<<grid2,block2>>>(d_coords,d_ij);
		//end

		//post-processing and optional cpu tour rearrangement
		if(iterations%REFRESH_ITER==0)
		{
			cudaDeviceSynchronize();
			getCudaDtype(d_differences,&bi.diff,1);
			getCudaInts(d_mink,&bi.k,1);
			int ij[2];
			getCudaInts(d_ij,ij,2);
			cudaDeviceSynchronize();
			//int ij[2];
			//ijfromk(bi.k,ij);
			if(bi.diff < 0)
			{
				//cpu tour rearrangement
				//swapDeviceTour(h_coords,d_coords,nc,ij);
				fprintf (stderr, "Iteration %d GPU min diff,i,j,raw k : %f,%d,%d,%d\n",iterations,bi.diff,ij[0],ij[1],bi.k);
			}
			else
			{
				fprintf (stderr, "No more improvements found! Stopping iteration.\n"); 
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

	//Check tour correctness.
	checkCoordValidity(h_coords,original,nc);
	//end

	//Output the timing and performance results.
	fprintf (stderr, "GPU min diff, raw k : %f, %d\n",bi.diff,bi.k);
	fprintf (stderr, "Execution time per iteration: %f ms\n", elapsedTimePerIteration);
	fprintf (stderr, "Speed (Gmoves/s): %f\n\n",N/(elapsedTimePerIteration/1000.0)/1e9);
	fprintf (stderr, "Threads Per Block: %d x %d\n",BX,BY);
	//fprintf (stderr, "Equivalent performance: %f GB/s\n", (N * sizeof (dtype) / elapsedTime) * 1e-6);
}
*/