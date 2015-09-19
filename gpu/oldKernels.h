#ifndef __OLDKERNELS_H__
#define __OLDKERNELS_H__

#include "distanceFunctions.h"

__device__ void warpReduce(volatile dtype*buffer,int tt)
{
		buffer[tt]+=buffer[tt+32];//__syncthreads();
		buffer[tt]+=buffer[tt+16];//__syncthreads();
		buffer[tt]+=buffer[tt+8];//__syncthreads();
		buffer[tt]+=buffer[tt+4];//__syncthreads();
		buffer[tt]+=buffer[tt+2];//__syncthreads();
		buffer[tt]+=buffer[tt+1];//__syncthreads();
}

__global__ void 
reduceMultAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	__shared__ dtype buffer[TPB];

	unsigned int i = ADDS * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int gs=gridDim.x*ADDS*blockDim.x;

	dtype sum = 0.0;

	while (i+5*TPB<N){
		sum+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB]+In[i+4*TPB]+In[i+5*TPB];//# ADDS should correspond here.
		//	buffer[threadIdx.x]+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB]+In[i+4*TPB]+In[i+5*TPB];//# ADDS should correspond here.
		i+=gs;
	}
	while (i+4*TPB<N){
		sum+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB]+In[i+4*TPB];//# ADDS should correspond here.
		//	buffer[threadIdx.x]+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB]+In[i+4*TPB];//# ADDS should correspond here.
		i+=gs;
	}
	while (i+3*TPB<N){
		sum+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB];//# ADDS should correspond here.
		//	buffer[threadIdx.x]+=In[i]+In[i+TPB]+In[i+2*TPB]+In[i+3*TPB];//# ADDS should correspond here.
		i+=gs;
	}
	while (i+2*TPB<N){
		sum+=In[i]+In[i+TPB]+In[i+2*TPB];//# ADDS should correspond here.
		//	buffer[threadIdx.x]+=In[i]+In[i+TPB]+In[i+2*TPB];//# ADDS should correspond here.
		i+=gs;
	}
	while (i+TPB<N){
		sum+=In[i]+In[i+TPB];//# ADDS should correspond here.
		//	buffer[threadIdx.x]+=In[i]+In[i+TPB];//# ADDS should correspond here.
		i+=gs;
	}
	while (i<N){
		sum+=In[i];
		//	buffer[threadIdx.x]+=In[i];
		i+=gs;
	}
	
	buffer[threadIdx.x]=sum;
	__syncthreads ();

	if(blockDim.x>=1024){ if (threadIdx.x < 512) { buffer[threadIdx.x] += buffer[threadIdx.x+512]; __syncthreads(); } }
	if(blockDim.x>=512){ if (threadIdx.x < 256) { buffer[threadIdx.x] += buffer[threadIdx.x+256]; __syncthreads(); } }
	if(blockDim.x>=256){ if (threadIdx.x < 128) { buffer[threadIdx.x] += buffer[threadIdx.x+128]; __syncthreads(); } }
	if(blockDim.x>=128){ if (threadIdx.x < 64) { buffer[threadIdx.x] += buffer[threadIdx.x+64]; __syncthreads(); } }
	if(threadIdx.x<32) warpReduce(buffer,threadIdx.x);
	// store back the reduced result 

	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}

__global__ void 
opt2_1_kernel (COORD *d_coords, dtype *d_differences, unsigned int nc,int globalStart)
{//Version 1. Naive implementation, using global memory.
	//One thread per swap.
	//d_coords are arranged in order of the current tour!
	//globalStart is to cope with the 1D block limit (e.g. 65535). Multiple kernels are called for all blocks
	
	//The following information is true for our 1D grid, 1D block.
	//total blocks is 'gridDim.x'
	//threads per block is 'blockDim.x'
	//total threads is 'blockDim.x*gridDim.x'
	//global thread id (gtid, it is a unique thread id) is 'blockIdx.x*blockDim.x+threadIdx.x' 
	//if threads perform more than one

	//gtid: global thread id
	int gtid = globalStart+blockIdx.x*blockDim.x+threadIdx.x;
	
	//compute i,j from gtid
	int i = (int)(((-1+sqrtf(1+4*2*gtid)))/2);//floating point calculation!
	int j = gtid-((i*(i+1))>>1);
	i=nc-3-i;
	j=nc-2-j;

	dtype difference = checkSwap_EUC_2D(
			d_coords[i-1].x,d_coords[i-1].y,d_coords[i].x,d_coords[i].y,
			d_coords[j].x,d_coords[j].y,d_coords[j+1].x,d_coords[j+1].y);
	
	d_differences[gtid]=difference;
}
__device__ void warpMinReduce(volatile dtype*sdata,int tid)
{
		sdata[tid] = (sdata[tid+32]<sdata[tid])?sdata[tid+32]:sdata[tid];
		sdata[tid] = (sdata[tid+16]<sdata[tid])?sdata[tid+16]:sdata[tid];
		sdata[tid] = (sdata[tid+8]<sdata[tid])?sdata[tid+8]:sdata[tid];
		sdata[tid] = (sdata[tid+4]<sdata[tid])?sdata[tid+4]:sdata[tid];
		sdata[tid] = (sdata[tid+2]<sdata[tid])?sdata[tid+2]:sdata[tid];
		sdata[tid] = (sdata[tid+1]<sdata[tid])?sdata[tid+1]:sdata[tid];
}
__global__ void 
opt2_2_kernel (COORD *d_coords, dtype *d_differences, unsigned int nc,int globalStart)
{//Version 2. Naive implementation, but with a min reduction step.
	//One thread per swap.
	//d_coords are arranged in order of the current tour!

	//__shared__ dtype buffer[TPB];
	
	//The following information is true for our 1D grid, 1D block.
	//total blocks is 'gridDim.x'
	//threads per block is 'blockDim.x'
	//total threads is 'blockDim.x*gridDim.x'
	//global thread id (gtid, it is a unique thread id) is 'blockIdx.x*blockDim.x+threadIdx.x' 
	//if threads perform more than one

	//gtid: global thread id
	int gtid = globalStart+blockIdx.x*blockDim.x+threadIdx.x;
	
	//compute i,j from gtid
	int i = (int)(((-1+sqrtf(1+4*2*gtid)))/2);//floating point calculation!
	int j = gtid-((i*(i+1))>>1);
	i=nc-3-i;
	j=nc-2-j;

	dtype difference = checkSwap_EUC_2D(
			d_coords[i-1].x,d_coords[i-1].y,d_coords[i].x,d_coords[i].y,
			d_coords[j].x,d_coords[j].y,d_coords[j+1].x,d_coords[j+1].y);
	
	//d_differences[gtid]=difference;

	//Reduction step
	__shared__ dtype sdata[TPB];
	unsigned int tid = threadIdx.x;
	sdata[tid] = difference;
	//unsigned int gridSize = TPB*2*gridDim.x;
	//unsigned int i = blockIdx.x*(TPB*2) + tid;
	//while (i < N) { sdata[tid] += g_idata[i] + g_idata[i+TPB]; i += gridSize; }
	__syncthreads();
	if (TPB >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[blockIdx.x] = sdata[0];
}
__global__ void 
opt2_3_kernel (COORD *d_coords, dtype *d_differences,
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 3. Transfer to shared memory first. 1D blockid, 2d threadid
	//One thread per swap.
	//d_coords are arranged in order of the current tour!
	//Assumes thread blocks are square!

	unsigned int ncoords = TPB2_3+1;//number/range of coordinates to copy for each i and j
	__shared__ COORD cc[2*(TPB2_3+1)];
	//i is the first TPB2 items, while j is in the 2nd TPB2 items.
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	unsigned int istart = bi*TPB2_3+1;
	int jstart =  (nc-2)-(bj+1)*TPB2_3+1;//bj*TPB2+2;
	unsigned int tid = threadIdx.x+threadIdx.y*TPB2_3;
	//Copy needed coords; only the first 2*(TPB2+1) threads are used.
	unsigned int maxn = nc-3;
	if(tid<ncoords)
	{//tx goes from [0, TPB2]
		int id=istart+tid-1;//copy start.
		if(id<maxn+1)
		{
			cc[tid]=d_coords[id];
		}
	}
	else if(tid<2*ncoords)
	{//tid goes from [TPB2+1, 2*TPB2+1]
		int id=jstart+tid-ncoords;
		//if(id<maxn+2)
		if(id<nc-bj*TPB2_3 && id>0)
		{
			cc[tid]=d_coords[id];
		}
	}
	int i = threadIdx.x+1;
	int j = threadIdx.y+ncoords;
	int gi = istart+threadIdx.x;
	int gj = jstart+threadIdx.y;
	__syncthreads();
	//gtid: global thread id
	//int gtid = blockIdx.x*blockDim.x+threadIdx.x;

	//compute i,j from gtid
	//int i = (int)(((-1+sqrtf(1+4*2*gtid)))/2);//floating point calculation!
	//int j = gtid-((i*(i+1))>>1);
	//i=nc-3-i;
	//j=nc-2-j;
	dtype difference = 0;
	//if(gi<maxn && gj<maxn)
	if(gi<maxn+1 && gj>gi && gj < nc-1-bj*TPB2_3)
	{
 		difference = checkSwap_EUC_2D(
			cc[i-1].x,cc[i-1].y,cc[i].x,cc[i].y,
			cc[j].x,cc[j].y,cc[j+1].x,cc[j+1].y);

	}
	__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2_3*TPB2_3];
	sdata[tid] = difference;
	//unsigned int gridSize = TPB*2*gridDim.x;
	//unsigned int i = blockIdx.x*(TPB*2) + tid;
	//while (i < N) { sdata[tid] += g_idata[i] + g_idata[i+TPB]; i += gridSize; }
	__syncthreads();
	if (TPB2_3*TPB2_3 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}
__global__ void 
opt2_31_kernel (COORD *d_coords, dtype *d_differences,
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 3. Seperate shared memory banks for i and j coordinates.
	//One thread per swap.
	//d_coords are arranged in order of the current tour!
	//Assumes thread blocks are square!

	unsigned int ncoords = TPB2_3+1;//number/range of coordinates to copy for each i and j
	//__shared__ COORD cc[2*(TPB2+1)];
	__shared__ dtype ix[TPB2_3+1];
	__shared__ dtype iy[TPB2_3+1];
	__shared__ dtype jx[TPB2_3+1];
	__shared__ dtype jy[TPB2_3+1];
	//i is the first TPB2 items, while j is in the 2nd TPB2 items.
	//compute bi,bj from blockIdx.x
	int block = globalStart+blockIdx.x;
	int bi = (int)(((-1+sqrtf(1+4*2*block)))/2);//floating point calculation!
	int bj = block-((bi*(bi+1))>>1);
	bi=BB-bi-1;
	bj=BB-bj-1-bi;
	unsigned int istart = bi*TPB2_3+1;
	int jstart =  (nc-2)-(bj+1)*TPB2_3+1;//bj*TPB2+2;
	unsigned int tid = threadIdx.x+threadIdx.y*TPB2_3;
	//Copy needed coords; only the first 2*(TPB2+1) threads are used.
	unsigned int maxn = nc-3;
	if(tid<ncoords)
	{//tx goes from [0, TPB2]
		int id=istart+tid-1;//copy start.
		if(id<maxn+1)
		{
			//cc[tid]=d_coords[id];
			//ii[tid]=d_coords[id];
			COORD coord = d_coords[id];
			ix[tid]=coord.x;
			iy[tid]=coord.y;
		}
	}
	else if(tid<2*ncoords)
	{//tid goes from [TPB2+1, 2*TPB2+1]
		int id=jstart+tid-ncoords;
		//if(id<maxn+2)
		if(id<nc-bj*TPB2_3 && id>0)
		{
			//cc[tid]=d_coords[id];
			//jj[tid-ncoords]=d_coords[id];
			COORD coord = d_coords[id];
			jx[tid-ncoords]=coord.x;
			jy[tid-ncoords]=coord.y;
		}
	}
	int i = threadIdx.x+1;
	int j = threadIdx.y;
	int gi = istart+threadIdx.x;
	int gj = jstart+threadIdx.y;
	__syncthreads();
	//gtid: global thread id
	//int gtid = blockIdx.x*blockDim.x+threadIdx.x;

	//compute i,j from gtid
	//int i = (int)(((-1+sqrtf(1+4*2*gtid)))/2);//floating point calculation!
	//int j = gtid-((i*(i+1))>>1);
	//i=nc-3-i;
	//j=nc-2-j;
	dtype difference = 0;
	//if(gi<maxn && gj<maxn)
	if(gi<maxn+1 && gj>gi && gj < nc-1-bj*TPB2_3)
	{
		//Suspected shared memory bank conflicts.
		dtype iprevx = ix[i-1];
		dtype iprevy = iy[i-1];
		dtype ixx = ix[i];
		dtype iyy = iy[i];
		dtype jxx = jx[j];
		dtype jyy = jy[j];
		dtype jnextx = jx[j+1];
		dtype jnexty = jy[j+1];
		difference = checkSwap_EUC_2D(
			iprevx,iprevy,ixx,iyy,
			jxx,jyy,jnextx,jnexty);
		//difference = checkSwap_EUC_2D(
		//	ix[i-1],iy[i-1],ix[i],iy[i],
		//	jx[j],jy[j],jx[j+1],jy[j+1]);
	}
	__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2_3*TPB2_3];
	sdata[tid] = difference;
	//unsigned int gridSize = TPB*2*gridDim.x;
	//unsigned int i = blockIdx.x*(TPB*2) + tid;
	//while (i < N) { sdata[tid] += g_idata[i] + g_idata[i+TPB]; i += gridSize; }
	__syncthreads();
	if (TPB2_3*TPB2_3 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2_3*TPB2_3 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}
__global__ void 
opt2_4_kernel (COORD *d_coords, dtype *d_differences,
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 4. Only registers! No shared memory except for reduction step.
	//d_coords are arranged in order of the current tour!
	//Assumes thread blocks are square!

	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];

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
	for(int c=0;c<TSD+1;++c)
	{
		int id=istart-1+c;
		if(id<maxn+1)
		{
			ii[c] = d_coords[id];
		}
	}
	for(int c=0;c<TSD+1;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj[c] = d_coords[id];
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	for(int sx=0;sx<TSD;++sx)
	{
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype difference = checkSwap_EUC_2D(
					ii[sx].x,ii[sx].y,ii[sx+1].x,ii[sx+1].y,
					jj[sy].x,jj[sy].y,jj[sy+1].x,jj[sy+1].y);
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	__shared__ dtype sdata[TPB2*TPB2];
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	__syncthreads();
	if (TPB2*TPB2 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}
__global__ void 
opt2_5_kernel (COORD *d_coords, dtype *d_differences,unsigned int nc,unsigned int BB,int globalStart)
{//Version 5, same as Version 4, but skipping the reduction step, just to test speed difference.
	//d_coords are arranged in order of the current tour!
	//Assumes thread blocks are square!

	COORD ii[(TSD+1)];//registers to hold coordinates
	COORD jj[(TSD+1)];

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
	for(int c=0;c<TSD+1;++c)
	{
		int id=istart-1+c;
		if(id<maxn+1)
		{
			ii[c] = d_coords[id];
		}
	}
	for(int c=0;c<TSD+1;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj[c] = d_coords[id];
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
	for(int sx=0;sx<TSD;++sx)
	{
		for(int sy=0;sy<TSD;++sy)
		{
			//first calculate the global indices to see if it is in calculation domain.
			int gi = istart+sx;
			int gj = jstart+sy;
			if(gi<maxn+1 && gj>gi && gj < nc-1-bj*BSD)
			{
				dtype difference = checkSwap_EUC_2D(
					ii[sx].x,ii[sx].y,ii[sx+1].x,ii[sx+1].y,
					jj[sy].x,jj[sy].y,jj[sy+1].x,jj[sy+1].y);
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();
	int tid = threadIdx.x+threadIdx.y*TPB2;
	int gtid = block*blockDim.x*blockDim.y+tid;
	d_differences[gtid] = min;
}
__global__ void 
opt2_6_kernel (COORD *d_coords, dtype *d_differences, unsigned int nc,unsigned int BB,int globalStart)
{//Version 6. An improvement on version 4.
	//Even though I allocate register arrays in v4, the compiler may store
	//it in local memory because the indices are not constant!
	//I try to unroll everything here.

	__shared__ dtype sdata[TPB2*TPB2];
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];

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
	for(int c=0;c<NCOORDS;++c)
	{
		int id=jstart+c;
		if(id<nc-bj*BSD && id>0)
		{
			jj[c] = d_coords[id];
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
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
				dtype difference = checkSwap_EUC_2D(
					ii[sx].x,ii[sx].y,ii[sx+1].x,ii[sx+1].y,
					jj[sy].x,jj[sy].y,jj[sy+1].x,jj[sy+1].y);
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	__syncthreads();
	if (TPB2*TPB2 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}
__global__ void 
opt2_7_kernel (COORD *d_coords, dtype *d_differences,unsigned int nc,unsigned int BB,int globalStart)
{//Version 7. Save distances for adjacent points in i and j (not between i and j points, though).

	__shared__ dtype sdata[TPB2*TPB2];
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];
	dtype savedi[TSD];//saved distances corresponding to i (i-1 to i)
	dtype savedj[TSD];//saved distances corresponding to j (j to j+1)

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
	//__syncthreads();

	//Now let us calculate the differences!
	dtype min = 0;
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
				dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
						distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				dtype difference = dnew-dold;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	__syncthreads();
	if (TPB2*TPB2 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}
__global__ void 
opt2_8_kernel (COORD *d_coords, dtype *d_differences, 
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 8. Save distances for adjacent points in i and j AND between points in i and j.

	__shared__ dtype sdata[TPB2*TPB2];
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
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	__syncthreads();
	if (TPB2*TPB2 >= 1024){ if (tid < 512) { sdata[tid] = (sdata[tid+512]<sdata[tid])?sdata[tid+512]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 512) { if (tid < 256) { sdata[tid] = (sdata[tid+256]<sdata[tid])?sdata[tid+256]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 256) { if (tid < 128) { sdata[tid] = (sdata[tid+128]<sdata[tid])?sdata[tid+128]:sdata[tid]; } __syncthreads(); }
	if (TPB2*TPB2 >= 128) { if (tid < 64) { sdata[tid] = (sdata[tid+64]<sdata[tid])?sdata[tid+64]:sdata[tid]; } __syncthreads(); }
	if (tid < 32) warpMinReduce(sdata,tid);
	if (tid == 0) d_differences[block] = sdata[0];
}











void cudaOpt2 (COORD *d_coords, unsigned int nc, int vv)
{
	//Set and determine thread and block dimensions
	int N = ((nc-3)*(nc-2))/2;//Number of tasks to do (each thread does >= 1)
	int NT = N;//Number of Threads.
	int NB = (NT + TPB - 1) / TPB;//Number of Blocks.
	int ND = -1;//Number of differences output from GPU to min-reduce. 
	int GX=1,GY=1;//block dimensions of grid.
	int BX=1,BY=1;//thread dimensions of block.
	int NN = nc-3;//such that N = NN*(NN+1)/2
	int BB = (NN+TPB2-1)/TPB2;//block dimension of the grid (TPB2-size square block)
	int SBB = TPB2*TSD;//Block dimension  terms of swaps. SBB = BB if swaps per thread is 1.
	int BB2 =(NN+SBB-1)/SBB;///Block dimension of the grid, for TPB2-sized square blocks WITH TSD^2 swaps performed per thread (versus regular 1).
	int gridSplits = -1;//for when the problem gets too large for 65535 blocks.
	int lastGX = -1;//if gridSplits > 0, this is the number of blocks in the last split (less than BLOCKLIMIT)
	int swapChecksPerBlock = -1;//number of swap checks handled per block.
	int totalBlocks = -1;
	int swapChecksPerThread = 1;
	switch(vv)
	{
		case 1:
			BX = TPB;
			ND = NT;
			GX = BLOCKLIMIT;
			totalBlocks = NB;
			swapChecksPerThread = 1;
			break;
		case 2:
			BX = TPB;
			ND = NB;
			GX = BLOCKLIMIT;
			totalBlocks = NB;
			swapChecksPerThread = 1;
			break;
		case 3:
			BB = (NN+TPB2_3-1)/TPB2_3;
			GX = BB*(BB+1)/2;
			//GY = 1;//(NN+TPB2-1)/TPB2;
			BX = TPB2_3;
			BY = TPB2_3;
			ND = GX;
			totalBlocks = BB*(BB+1)/2;
			swapChecksPerThread = 1;
			break;
		case 31:
			BB = (NN+TPB2_3-1)/TPB2_3;
			GX = BB*(BB+1)/2;
			//GY = 1;//(NN+TPB2-1)/TPB2;
			BX = TPB2_3;
			BY = TPB2_3;
			ND = GX;
			totalBlocks = BB*(BB+1)/2;
			swapChecksPerThread = 1;
			break;
		case 4:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		case 5:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = TPB2*TPB2*GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		case 6:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		case 7:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		case 8:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		case 9:
			GX = BB2*(BB2+1)/2;
			BX = TPB2;
			BY = TPB2;
			ND = GX;
			totalBlocks = BB2*(BB2+1)/2;
			swapChecksPerThread = TSD*TSD;
			break;
		default:
			break;
	}
	swapChecksPerBlock = BX*BY*swapChecksPerThread;
	gridSplits = totalBlocks/BLOCKLIMIT;

	GX=BLOCKLIMIT;
	dim3 grid (GX,GY);
	dim3 block (BX,BY);
	lastGX = totalBlocks-gridSplits*BLOCKLIMIT;
	dim3 lastGrid (lastGX,GY);

	//Output information about run time parameters.
	fprintf (stderr, "Kernel Version %d\n",vv);
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
	dtype *d_differences;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences, ND * sizeof (dtype)));
	int *h_mink = new int[ND]; 
	int *d_mink;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink, ND * sizeof (int)));
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);//Shared Memory <-> L1 Cache setting.

	//Timer initialization and start
	cudaEvent_t start, stop;
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	//

	//EXECUTE KERNEL
	//Run the kernel for some iterations
	BESTIMPROVEMENT bi = {0,0};
	int startSwap=-1;
	for(int i = 0; i < NUM_ITER; i++) {
		switch(vv)
		{
			case 1:
				for(int j=0;j<gridSplits;++j)
				{
					startSwap = swapChecksPerBlock*j*BLOCKLIMIT;
					opt2_1_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,startSwap);
				}
				startSwap = swapChecksPerBlock*gridSplits*BLOCKLIMIT;
				opt2_1_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,startSwap);
				break;
			case 2:
				for(int j=0;j<gridSplits;++j)
				{
					startSwap = swapChecksPerBlock*j*BLOCKLIMIT;
					opt2_2_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,startSwap);
				}
				startSwap = swapChecksPerBlock*gridSplits*BLOCKLIMIT;
				opt2_2_kernel <<<lastGrid, block>>> (d_coords, d_differences,
						nc,startSwap);
				break;
			case 3:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_3_kernel <<<grid, block>>> (d_coords, d_differences, 
						nc,BB,j*BLOCKLIMIT);
				}
				opt2_3_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB,gridSplits*BLOCKLIMIT);
				break;
			case 31:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_31_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB,j*BLOCKLIMIT);
				}
				opt2_31_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB,gridSplits*BLOCKLIMIT);
				break;
			case 4:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_4_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB2,j*BLOCKLIMIT);
				}
				opt2_4_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			case 5:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_5_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB2,j*BLOCKLIMIT);
				}
				opt2_5_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			case 6:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_6_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB2,j*BLOCKLIMIT);
				}
				opt2_6_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			case 7:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_7_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB2,j*BLOCKLIMIT);
				}
				opt2_7_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			case 8:
				for(int j=0;j<gridSplits;++j)
				{
					opt2_8_kernel <<<grid, block>>> (d_coords, d_differences,
						nc,BB2,j*BLOCKLIMIT);
				}
				opt2_8_kernel <<<lastGrid, block>>> (d_coords, d_differences,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			case 9:
				for(int j=0;j<gridSplits;++j)
				{
					kernel9 <<<grid, block>>> (d_coords, d_differences,d_mink,
						nc,BB2,j*BLOCKLIMIT);
				}
				kernel9 <<<lastGrid, block>>> (d_coords, d_differences,d_mink,
					nc,BB2,gridSplits*BLOCKLIMIT);
				break;
			default:
				break;
		}
		cudaDeviceSynchronize();
		if (SEQUENTIAL==1)
		{
			if(vv<9)
			{
				bi = minReduceCpu(h_differences,d_differences,ND);
			}
			else
			{
				bi = minReduceCpu2(h_differences,d_differences,h_mink,d_mink,ND);
			}
		}
	}
	//END KERNEL EXECUTION

	//Timer stop, get time, destroy timer resources
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	float elapsedTime;
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime /= NUM_ITER;
	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

	//Perform the reduction outside of loop if desired.
	if (SEQUENTIAL==0)
	{
		if(vv<9)
		{
			bi = minReduceCpu(h_differences,d_differences,ND);
		}
		else
		{
			bi = minReduceCpu2(h_differences,d_differences,h_mink,d_mink,ND);
		}
		delete h_differences;
		delete h_mink;
	}
	//End outside loop reduction.

	//Output the timing and performance results.
	fprintf (stderr, "GPU min diff, raw k : %f, %d\n",bi.diff,bi.k);
	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Speed (Gmoves/s): %f\n\n",N/(elapsedTime/1000.0)/1e9);
	//fprintf (stderr, "Equivalent performance: %f GB/s\n", (N * sizeof (dtype) / elapsedTime) * 1e-6);
}








void cudaOpt2 (COORD *d_coords, unsigned int nc)
{
	//Set and determine thread and block dimensions
	int N = ((nc-3)*(nc-2))/2;//Number of tasks to do (each thread does >= 1)
	int NT = N;//Number of Threads.
	int NB = (NT + TPB - 1) / TPB;//Number of Blocks.
	int ND = -1;//Number of differences output from GPU to min-reduce. 
	int GX=1,GY=1;//block dimensions of grid.
	int BX=1,BY=1;//thread dimensions of block.
	int NN = nc-3;//such that N = NN*(NN+1)/2
	int BB = (NN+TPB2-1)/TPB2;//block dimension of the grid (TPB2-size square block)
	int SBB = TPB2*TSD;//Block dimension  terms of swaps. SBB = BB if swaps per thread is 1.
	int BB2 =(NN+SBB-1)/SBB;///Block dimension of the grid, for TPB2-sized square blocks WITH TSD^2 swaps performed per thread (versus regular 1).
	int gridSplits = -1;//for when the problem gets too large for 65535 blocks.
	int lastGX = -1;//if gridSplits > 0, this is the number of blocks in the last split (less than BLOCKLIMIT)
	int swapChecksPerBlock = -1;//number of swap checks handled per block.
	int totalBlocks = -1;
	int swapChecksPerThread = 1;
	
		GX = BB2*(BB2+1)/2;
		BX = TPB2;
		BY = TPB2;
		ND = GX;
		totalBlocks = BB2*(BB2+1)/2;
		swapChecksPerThread = TSD*TSD;
	
	swapChecksPerBlock = BX*BY*swapChecksPerThread;
	gridSplits = totalBlocks/BLOCKLIMIT;

	GX=BLOCKLIMIT;
	dim3 grid (GX,GY);
	dim3 block (BX,BY);
	lastGX = totalBlocks-gridSplits*BLOCKLIMIT;
	dim3 lastGrid (lastGX,GY);

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
	dtype *d_differences;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_differences, ND * sizeof (dtype)));
	int *h_mink = new int[ND]; 
	int *d_mink;
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_mink, ND * sizeof (int)));
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);//Shared Memory <-> L1 Cache setting.

	//Timer initialization and start
	cudaEvent_t start, stop;
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	//

	//EXECUTE KERNEL
	//Run the kernel for some iterations
	BESTIMPROVEMENT bi = {0,0};
	int startSwap=-1;
	for(int i = 0; i < NUM_ITER; i++) {
		for(int j=0;j<gridSplits;++j)
		{
			kernel9 <<<grid, block>>> (d_coords, d_differences,d_mink,
				nc,BB2,j*BLOCKLIMIT);
		}
		kernel9 <<<lastGrid, block>>> (d_coords, d_differences,d_mink,
			nc,BB2,gridSplits*BLOCKLIMIT);
		cudaDeviceSynchronize();
		if (SEQUENTIAL==1)
		{
			bi = minReduceCpu2(h_differences,d_differences,h_mink,d_mink,ND);
		}
	}
	//END KERNEL EXECUTION

	//Timer stop, get time, destroy timer resources
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	float elapsedTime;
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime /= NUM_ITER;
	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

	//Perform the reduction outside of loop if desired.
	if (SEQUENTIAL==0)
	{
		bi = minReduceCpu2(h_differences,d_differences,h_mink,d_mink,ND);
		delete h_differences;
		delete h_mink;
	}
	//End outside loop reduction.

	//Output the timing and performance results.
	fprintf (stderr, "GPU min diff, raw k : %f, %d\n",bi.diff,bi.k);
	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Speed (Gmoves/s): %f\n\n",N/(elapsedTime/1000.0)/1e9);
	//fprintf (stderr, "Equivalent performance: %f GB/s\n", (N * sizeof (dtype) / elapsedTime) * 1e-6);
}



BESTIMPROVEMENT minReduceCpu(dtype*h,dtype*d,int n)
{
	CUDA_CHECK_ERROR (cudaMemcpy (h, d, n * sizeof (dtype), cudaMemcpyDeviceToHost));
	//A minimum reduction.
	dtype minDifference=0;
	int minI = -1;
	for(int i=0; i<n; i++)
	{
		if(h[i] < minDifference)
		{
			minDifference = h[i];
			minI = i;
		}
	}
	BESTIMPROVEMENT ret = {minI,minDifference};
	return ret;
}

__device__ void warpMinReduce2_int(volatile int*sdata,volatile int*smink,int tid)
{
	int currentsdata = sdata[tid];
	int currentsmink = smink[tid];
	int nextsdata = sdata[tid+32];
	int nextsmink = smink[tid+32];
	int check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+16];
	nextsmink = smink[tid+16];
	check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+8];
	nextsmink = smink[tid+8];
	check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+4];
	nextsmink = smink[tid+4];
	check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+2];
	nextsmink = smink[tid+2];
	check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+1];
	nextsmink = smink[tid+1];
	check = nextsdata<currentsdata;
	smink[tid] = currentsmink = (check)?nextsmink:currentsmink;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
}

__global__ void 
kernel9_int (COORD *d_coords, dtype *d_differences,int *d_mink, 
	unsigned int nc,unsigned int BB,int globalStart)
{//Version 9. Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.
	COORD ii[NCOORDS];//registers to hold coordinates
	COORD jj[NCOORDS];
	int savedi[TSD];//saved distances corresponding to i (i-1 to i)
	int savedj[TSD];//saved distances corresponding to j (j to j+1)
	int savedij[TSD+1][TSD+1];//saved distances (i-1 -> j)

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
		savedi[i]=distanceEUC_2D_int(ii[i].x,ii[i].y,ii[i+1].x,ii[i+1].y);
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
		savedj[j]=distanceEUC_2D_int(jj[j].x,jj[j].y,jj[j+1].x,jj[j+1].y);
	}
	#pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D_int(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y);
		}
	}
	//__syncthreads();

	//Now let us calculate the differences!
	int min = 0;
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
				int dold = savedi[sx]+savedj[sy];
				int dnew = savedij[sx][sy]+savedij[sx+1][sy+1];
				//dtype dnew = distanceEUC_2D(ii[sx].x,ii[sx].y,jj[sy].x,jj[sy].y)+
				//		distanceEUC_2D(ii[sx+1].x,ii[sx+1].y,jj[sy+1].x,jj[sy+1].y);
				int difference = dnew-dold;
				mink = (difference<min)?(kfromij(gi,gj)):mink;
				min = (difference<min)?difference:min;
			}
		}
	}
	//__syncthreads();

	//Reduction step
	__shared__ int sdata[TPB2*TPB2];
	__shared__ int smink[TPB2*TPB2];
	int tid = threadIdx.x+threadIdx.y*TPB2;
	sdata[tid] = min;
	smink[tid] = mink;
	__syncthreads();
	int check=-1;
	int currentsdata = sdata[tid];
	int nextsdata;
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
	if (tid < 32) warpMinReduce2_int(sdata,smink,tid);
	if (tid == 0){
		d_differences[block] = sdata[0];
		d_mink[block] = smink[0];
	}
}





#endif