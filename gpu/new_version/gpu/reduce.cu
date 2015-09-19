#include "types.hh"
#include "parameters.hh"

// __device__ inline int ij2k(int i,int j)
// {
// 	i-=1;
// 	j-=2;
// 	return i+((j*(j+1))>>1);
// }

// __device__ inline void k2ij(int k,int *i, int *j)
// {
// 	int i_ = (int)(((-1+sqrtf(1+4*2*k)))/2);//floating point calculation!
// 	int j_ = k-((i_*(i_+1))>>1);
// 	*i = j_+1;
// 	*j = i_+2;
// }

// __device__ void warpMinReduce2(volatile dtype*sdata,volatile int*smink,int tid)
// {
// 	dtype currentsdata = sdata[tid];
// 	int currentsmink = smink[tid];
// 	dtype nextsdata = sdata[tid+32];
// 	int nextsmink = smink[tid+32];
// 	int check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 	nextsdata = sdata[tid+16];
// 	nextsmink = smink[tid+16];
// 	check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 	nextsdata = sdata[tid+8];
// 	nextsmink = smink[tid+8];
// 	check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 	nextsdata = sdata[tid+4];
// 	nextsmink = smink[tid+4];
// 	check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 	nextsdata = sdata[tid+2];
// 	nextsmink = smink[tid+2];
// 	check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 	nextsdata = sdata[tid+1];
// 	nextsmink = smink[tid+1];
// 	check = nextsdata<currentsdata;
// 	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// }
// __global__ void
// gpuSwapReduce (dtype *d_differences, dtype *d_differences2, 
// 	int *d_mink, int *d_mink2, int *d_ij, int nd)
// {//Reduce to find minimum difference.
// 	//Then use the index to get the minimum k-index.
// 	__shared__ dtype sdata[REDUCE_TPB];
// 	__shared__ int smink[REDUCE_TPB];

// 	int tid = threadIdx.x;
// 	int gid = threadIdx.x+blockIdx.x*blockDim.x;
// 	if (gid < nd)
// 	{
// 		sdata[tid] = d_differences[gid];
// 		smink[tid] = d_mink[gid];
// 	}
// 	else
// 	{
// 		sdata[tid] = 0;
// 		smink[tid] = 0;
// 	}
// 	__syncthreads();
// 	int check=-1;
// 	dtype currentsdata = sdata[tid];
// 	dtype nextsdata;
// 	int currentsmink = smink[tid];
// 	int nextsmink;
// 	if (REDUCE_TPB >= 1024){ if (tid < 512) {
// 		//check = sdata[tid+512]<sdata[tid];
// 		nextsdata = sdata[tid+512];
// 		nextsmink = smink[tid+512];
// 		check = nextsdata<currentsdata;
// 		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
// 		__syncthreads(); }
// 	if (REDUCE_TPB >= 512) { if (tid < 256) {
// 		nextsdata = sdata[tid+256];
// 		nextsmink = smink[tid+256];
// 		check = nextsdata<currentsdata;
// 		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
// 		__syncthreads(); }
// 	if (REDUCE_TPB >= 256) { if (tid < 128) {
// 		nextsdata = sdata[tid+128];
// 		nextsmink = smink[tid+128];
// 		check = nextsdata<currentsdata;
// 		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
// 		__syncthreads(); }
// 	if (REDUCE_TPB >= 128) { if (tid < 64) {
// 		nextsdata = sdata[tid+64];
// 		nextsmink = smink[tid+64];
// 		check = nextsdata<currentsdata;
// 		sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
// 		smink[tid] = currentsmink = (check)?nextsmink:currentsmink; } 
// 		__syncthreads(); }
// 	if (tid < 32) warpMinReduce2(sdata,smink,tid);
// 	if (tid == 0){
// 		d_differences2[blockIdx.x] = sdata[0];
// 		d_mink2[blockIdx.x] = smink[0];
// 	}
// 	if((gridDim.x == 1) and (tid == 0))
// 	{
// 		k2ij(smink[0],d_ij,d_ij+1);
// 	}
// }

// __global__ void
// correctReduceBuffer (dtype *d_differences,dtype *d_differences2,int *d_mink,int *d_mink2)
// {
// 	int tid = threadIdx.x + blockDim.x*blockIdx.x;
// 	if(tid==0)
// 	{
// 		d_differences[0] = d_differences2[0];
// 		d_mink[0] = d_mink2[0];
// 	}
// }