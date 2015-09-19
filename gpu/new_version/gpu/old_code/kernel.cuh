__device__ inline int kfromij(int i,int j)
{
	i-=1;
	j-=2;
	return i+((j*(j+1))>>1);
}
__device__ void warpMinReduce(volatile dtype*sdata,int tid)
{
	dtype currentsdata = sdata[tid];
	dtype nextsdata = sdata[tid+32];
	int check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+16];
	check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+8];
	check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+4];
	check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+2];
	check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
	nextsdata = sdata[tid+1];
	check = nextsdata<currentsdata;
	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata;
}
// __device__ void warpMinReduce(volatile dtype*sdata,int tid);
__device__ void warpMinReduce2(volatile dtype*sdata,volatile int*smink,int tid);

__global__ void 
kernel(const dtype *d_x, const dtype *d_y, dtype *d_diff,int *d_k, 
	const unsigned int nc, const unsigned int BB, const int globalStart)
{//Version 9. Save distances for adjacent points in i and j AND between points in i and j.
	//also get the index.


	// COORD ii[NCOORDS];//registers to hold coordinates
	dtype ii_x[NCOORDS];
	dtype ii_y[NCOORDS];
	// COORD jj[NCOORDS];
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
	for(int i=0;i<TSD;++i)
	{
		savedi[i]=distanceEUC_2D(ii_x[i],ii_y[i],ii_x[i+1],ii_y[i+1]);
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
	#pragma unroll
	for(int j=0;j<TSD;++j)
	{
		savedj[j]=distanceEUC_2D(jj_x[j],jj_y[j],jj_x[j+1],jj_y[j+1]);
	}
	#pragma unroll
	for(int sx=0;sx<TSD+1;++sx)
	{
		#pragma unroll
		for(int sy=0;sy<TSD+1;++sy)
		{
			savedij[sx][sy]=distanceEUC_2D(ii_x[sx],ii_y[sx],jj_x[sy],jj_y[sy]);
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

	//for cub version

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

	// //Reduction step
	// __shared__ dtype sdata[TPB2*TPB2];
	// __shared__ dtype min_block;
	// int tid = threadIdx.x+threadIdx.y*TPB2;
	// sdata[tid] = min;
	// dtype currentsdata = min;//sdata[tid];
	// int check=-1;
	// dtype nextsdata;
	// __syncthreads();
	// if (TPB2*TPB2 >= 1024){ if (tid < 512) {
	// 	//check = sdata[tid+512]<sdata[tid];
	// 	nextsdata = sdata[tid+512];
	// 	check = nextsdata<currentsdata;
	// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata; }
	// 	__syncthreads(); }
	// if (TPB2*TPB2 >= 512) { if (tid < 256) {
	// 	nextsdata = sdata[tid+256];
	// 	check = nextsdata<currentsdata;
	// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata; }
	// 	__syncthreads(); }
	// if (TPB2*TPB2 >= 256) { if (tid < 128) {
	// 	nextsdata = sdata[tid+128];
	// 	check = nextsdata<currentsdata;
	// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata; }
	// 	__syncthreads(); }
	// if (TPB2*TPB2 >= 128) { if (tid < 64) {
	// 	nextsdata = sdata[tid+64];
	// 	check = nextsdata<currentsdata;
	// 	sdata[tid] = currentsdata = (check)?nextsdata:currentsdata; }
	// 	__syncthreads(); }
	// if (tid < 32) warpMinReduce(sdata,tid);
	// if (tid == 0){
	// 	min_block = sdata[0];
	// }
	// __syncthreads();
	// if (min_block == min)
	// {
	// 	d_k[block] = mink;
	// 	d_diff[block] = min;
	// }
}