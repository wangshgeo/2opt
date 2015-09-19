		
		//gpu tour rearrangement
		cudaDeviceSynchronize();
		//fprintf(stderr,"grid2, block2: %d, %d\n",cityblocks,SWAP_TPB);
		gpuSwap2<<<grid2,block2>>>(d_x,d_y,d_ij);
		//end

		//OLD REDUCTION VERSION
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
		// END OLD VERSION