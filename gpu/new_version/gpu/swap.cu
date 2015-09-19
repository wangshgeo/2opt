#include "types.hh"
#include "cub-1.4.1/cub/cub.cuh"

__device__ inline void k2ij(int k,int *i, int *j)
{
	int i_ = (int)(((-1+sqrtf(1+4*2*k)))/2);//floating point calculation!
	int j_ = k-((i_*(i_+1))>>1);
	*i = j_+1;
	*j = i_+2;
}
__device__ inline int ij2k(int i,int j)
{
	i-=1;
	j-=2;
	return i+((j*(j+1))>>1);
}
// void
// gpuSwap (COORD *d_tour,int*d_ij)
// {//The i,j indices are known beforehand.
// 	//and the blocks are configured knowing i,j (require host knowledge)
// 	int i = d_ij[0];
// 	int j = d_ij[1];
// 	int diff = j-i+1;
// 	int li = threadIdx.x + blockIdx.x*blockDim.x;
// 	int gi1 = i + li;
// 	int gi2 = j - li;
// 	if(li <= diff>>1)
// 	{
// 		COORD tmp1 = d_tour[gi1];
// 		COORD tmp2 = d_tour[gi2];
// 		d_tour[gi1] = tmp2;
// 		d_tour[gi2] = tmp1;
// 	}
// }
__global__ void
gpuSwap2 (dtype *d_x, dtype* d_y, int* d_ij)
{//The i,j indices are known beforehand.
	//blocks are NOT configured knowing i,j (which requires host knowledge)
	int gi = threadIdx.x + blockIdx.x*blockDim.x;
	int i = d_ij[0];
	int j2;
	if(gi >= i)
	{
		int j = d_ij[1];
		int diff = j-i+1;
		j2 = i+(diff>>1);
		if(gi <= j2)
		{
			int li = gi - i;
			int gi2 = j - li;
			// COORD tmp1 = d_tour[gi];
			dtype tmp1_x = d_x[gi];
			dtype tmp1_y = d_y[gi];
			// COORD tmp2 = d_tour[gi2];
			dtype tmp2_x = d_x[gi2];
			dtype tmp2_y = d_y[gi2];
			// d_tour[gi] = tmp2;
			d_x[gi] = tmp2_x;
			d_y[gi] = tmp2_y;
			// d_tour[gi2] = tmp1;
			d_x[gi2] = tmp1_x;
			d_y[gi2] = tmp1_y;
			/*
			if(gi==j2)
			{
				d_ij[0] = -42;
				d_ij[1] = -42;
			}
			*/
		}
	}
}
__global__ void
gpuSwap3 (dtype *d_x, dtype* d_y, cub::KeyValuePair<int, dtype> *d_out, dtype* d_differences, int* d_k,int*d_ij)
{//The i,j indices are known beforehand.
	//blocks are NOT configured knowing i,j (which requires host knowledge)
	int mink_index = d_out->key;
	int mink = d_k[mink_index];
	dtype mincost = d_out->value;
	d_differences[0] = mincost;
	d_k[0] = mink;
	int ij[2];
	k2ij(mink,ij,ij+1);
	d_ij[0] = ij[0];
	d_ij[1] = ij[1];
	int gi = threadIdx.x + blockIdx.x*blockDim.x;
	int i = ij[0];
	int j2;
	if(gi >= i)
	{
		int j = ij[1];
		int diff = j-i+1;
		j2 = i+(diff>>1);
		if(gi <= j2)
		{
			int li = gi - i;
			int gi2 = j - li;
			// COORD tmp1 = d_tour[gi];
			dtype tmp1_x = d_x[gi];
			dtype tmp1_y = d_y[gi];
			// COORD tmp2 = d_tour[gi2];
			dtype tmp2_x = d_x[gi2];
			dtype tmp2_y = d_y[gi2];
			// d_tour[gi] = tmp2;
			d_x[gi] = tmp2_x;
			d_y[gi] = tmp2_y;
			// d_tour[gi2] = tmp1;
			d_x[gi2] = tmp1_x;
			d_y[gi2] = tmp1_y;
			/*
			if(gi==j2)
			{
				d_ij[0] = -42;
				d_ij[1] = -42;
			}
			*/
		}
	}
}
__global__ void
gpuSwap4 (dtype *d_x, dtype* d_y, 
	cub::KeyValuePair<int, dtype> *d_out, 
	dtype* d_differences, int* d_k,
	int* d_i,int *d_j,
	cost_t *d_segment_lengths)
{
	int min_index = d_out->key;
	int ij[2];
	ij[0] = d_i[min_index];
	ij[1] = d_j[min_index];
	dtype mincost = d_out->value;
	d_differences[0] = mincost;
	d_k[0] = ij2k(ij[0],ij[1]);

	int gi = threadIdx.x + blockIdx.x*blockDim.x;
	int i = ij[0];
	int j2;
	if(gi >= i)
	{
		int j = ij[1];
		int diff = j-i+1;
		j2 = i+(diff>>1);
		if(gi <= j2)
		{
			int li = gi - i;
			int gi2 = j - li;
			// COORD tmp1 = d_tour[gi];
			dtype tmp1_x = d_x[gi];
			dtype tmp1_y = d_y[gi];
			cost_t tmp1_sl = d_segment_lengths[gi];
			// COORD tmp2 = d_tour[gi2];
			dtype tmp2_x = d_x[gi2];
			dtype tmp2_y = d_y[gi2];
			cost_t tmp2_sl = d_segment_lengths[gi2];
			// d_tour[gi] = tmp2;
			d_x[gi] = tmp2_x;
			d_y[gi] = tmp2_y;
			d_segment_lengths[gi] = tmp2_sl;
			// d_tour[gi2] = tmp1;
			d_x[gi2] = tmp1_x;
			d_y[gi2] = tmp1_y;
			d_segment_lengths[gi2] = tmp1_sl;
			/*
			if(gi==j2)
			{
				d_ij[0] = -42;
				d_ij[1] = -42;
			}
			*/
		}
	}
}

__global__ void
updateLengths (dtype *d_x, dtype* d_y, 
	cub::KeyValuePair<int, dtype> *d_out, 
	int* d_i,int *d_j,
	cost_t *d_segment_lengths)
{
	if(threadIdx.x == 0 or threadIdx.y == 1)
	{
		int min_index = d_out->key;
		int k = (threadIdx.x == 0) ? d_i[min_index] : d_j[min_index];

		dtype dx = d_x[k+1] - d_x[k];
		dtype dy = d_y[k+1] - d_y[k];
		d_segment_lengths[k] = sqrt(dx*dx + dy*dy);
	}
}