//Indexing tests for kernel 4.

#include <iostream>
#include <math.h>
using namespace std;

#define TSD 3//thread swap dimension (swaps per thread: TSD^2)

#define NC 14 //Number of cities
#define TPB2 2 //Threads per block for kernels 3+

#define MAXN ((NC)-3) //max dimension of the triangle swap map.
#define N (((NC-3)*(NC-2))/2) // total number of swaps.
#define SBB ((TPB2)*(TSD)) //Block dimension in terms of swaps. SBB = BB if swaps per thread is 1.
#define BB2 (((MAXN)+(SBB)-1)/(SBB)) //grid dimension (blocks)
#define NB ((BB2)*((BB2)+1)/2) //number of blocks

#define BSD ((TPB2)*(TSD)) //Block swap dimension 

void printk();
void printControl();
void printArray(int arr[],int len);

int d_coords[NC];

int main()
{
	printk();

	printControl();

	//This loop simulates the blocks.
	for(int b=0;b<NB;++b)
	{
		//This double loop simulates the thread block.
		cout << "Block " << b << ":\n";
		for(int tx=0;tx<TPB2;++tx)
		{
			for(int ty=0;ty<TPB2;++ty)
			{

				//THIS POINT SIMULATES A THREAD.
				cout << "Thread " << tx << "," << ty << "\n";

				int ii[(TSD+1)];
				int jj[(TSD+1)];
				for(int i=0;i<TSD+1;++i)
				{
					ii[i]=-1;
					jj[i]=-1;
				}

				//compute bi,bj from blockIdx.x
				int bi = (int)(((-1+sqrtf(1+4*2*b)))/2);//floating point calculation!
				int bj = b-((bi*(bi+1))>>1);
				bi=BB2-bi-1;
				bj=BB2-bj-1-bi;
				unsigned int istart = bi*BSD+1;//i start from block
				istart+=TSD*tx;//i start from thread
				int jstart =  (NC-2)-(bj+1)*BSD+1;//j start from block
				jstart+=TSD*ty;//j start from thread

				//Now transfer the memory from global into registers.
				for(int c=0;c<TSD+1;++c)
				{
					int id=istart-1+c;
					if(id<MAXN+1)
					{
						ii[c] = d_coords[id];
					}
				}
				for(int c=0;c<TSD+1;++c)
				{
					int id=jstart+c;
					if(id<NC-bj*BSD && id>0)
					{
						jj[c] = d_coords[id];
					}
				}

				//Now let us calculate the differences!
				float min = 0;
				for(int sx=0;sx<TSD;++sx)
				{
					int syflag = 0;
					for(int sy=TSD-1;sy>=0;--sy)
					{
						//first calculate the global indices to see if it is in calculation domain.
						int gi = istart+sx;
						int gj = jstart+sy;
						if(gi<MAXN+1 && gj>gi && gj < NC-1-bj*BSD)
						{
							/*
							dtype difference = checkSwap_EUC_2D(
								ii[sx].x,ii[sx].y,ii[sx+1].x,ii[sx+1].y,
								jj[sy].x,jj[sy].y,jj[sy+1].x,jj[sy+1].y);
							min = (difference<min)?difference:min;
							*/
							min=1;
							syflag=1;
							//cout << gi << "," << gj << "   ";
							cout << gi << "," << gj << "(" << sx << "," << sy << ")" << "   ";
						}
					}
					if(syflag)
						cout << "\n";
				}
				if(min>0)
				{
					cout << "ii: ";
					printArray(ii,TSD+1);
					cout << "\n";
					cout << "jj: ";
					printArray(jj,TSD+1);
					cout << "\n";
				}


				//END THREAD SIMULATION.

			}
		}
	}


	return 0;
}


void printControl()
{
	//coordinates
	for(int i=0;i<NC;++i)
	{
		d_coords[i] = i;	
	}
	cout << "\nprintControl(): \n";
	cout << "Coordinates: ";
	printArray(d_coords,NC);
	cout << "\n";
	//i,j
	for(int i=1;i<NC-2;++i)
	{
		if((i-1)%BSD==0 and i!=1)
			cout << "\n";
		cout << "\n";
		for(int j=NC-2;j>i;--j)
		{
			if(((NC-2)-j)%BSD==0 and j!=NC-2)
				cout << "   ";
			cout << i << "," << j << "   ";
		}
	}
	cout << "\n\n\n";
}
void printk()
{
	//coordinates
	for(int i=0;i<NC;++i)
	{
		d_coords[i] = i;	
	}
	cout << "\nprintk(): \n";
	//i,j
	for(int i=1;i<NC-2;++i)
	{
		cout << "\n";
		for(int j=NC-2;j>i;--j)
		{
			int ki = i-1;
			int kj = j-2;
			int k = ki+((kj*(kj+1))>>1);
			cout << k << "\t";
		}
	}
	cout << "\n\n\n";
}
void printArray(int arr[],int len)
{
	for (int i=0;i<len;++i)
	{
		if(arr[i]<0) break;
		cout << arr[i] << "   ";
	}
}